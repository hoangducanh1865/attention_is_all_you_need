import os
import torch
import numpy as np
from transformers import AutoTokenizer,get_scheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_from_disk
from tqdm import tqdm
from src.transformer import Transformer
from src.data_manager import DataManager # To get DataManager.translation_collator
from src.tokenizer import FrenchTokenizer
from src.config import Config,TransformerConfig
from src.utils import translation_collator
os.environ['TOKENIZERS_PARALLELISM']='false'
class Trainer:
    def __init__(self):
        self.training_config=Config.training_config
        self.src_tokenizer = AutoTokenizer.from_pretrained(Config.PRETRAINED_ENGLISH_TOKENIZER_NAME)
        self.tgt_tokenizer = FrenchTokenizer(Config.TRAINED_FRENCH_TOKENIZER_PATH)
        self.tokenized_dataset_dir=Config.SAVE_TOK_DIR # Tokenized dataset directory
        self.experiment_dir=os.path.join(Config.training_config .working_dir,Config.training_config.experiment_name)
        self.accelerator=Accelerator(project_dir=self.experiment_dir,
                                     log_with='wandb')
        '''self.accelerator.init_trackers(self.config.experiment_name)'''
        self.config=Config.transformer_config
        self.dataset=load_from_disk(self.tokenized_dataset_dir)
        self.accelerator.print(self.dataset)
        self.collate_fn=translation_collator(src_tokenizer=self.src_tokenizer,tgt_tokenizer=self.tgt_tokenizer)
        self.minibatch_size=Config.training_config.batch_size // Config.training_config.gradient_accumulation_steps # @QUESTION: What is the idea behind this scale step?
        self.train_dataloader=DataLoader(self.dataset['train'],
                                         batch_size=self.minibatch_size,
                                         num_workers=Config.training_config.num_workers,
                                         collate_fn=self.collate_fn,
                                         shuffle=True)
        self.test_dataloader=DataLoader(self.dataset['test'],
                                        batch_size=self.minibatch_size,
                                        num_workers=Config.training_config.num_workers,
                                        collate_fn=self.collate_fn,
                                        shuffle=True)
        self.model=Transformer(self.config)
        self.accelerator.print(f'Number of parameters: {self.get_num_parameters()}')
        self.optimizer=torch.optim.AdamW(params=self.model.parameters(),
                                         lr=Config.training_config.learning_rate,
                                         betas=Config.training_config.betas,
                                         eps=Config.training_config.adam_eps)
        self.scheduler=get_scheduler(name=Config.training_config.scheduler_type,
                                     optimizer=self.optimizer,
                                     num_warmup_steps=Config.training_config.num_warmup_steps * self.accelerator.num_processes, # Since every operations happen multiple times so we need to scale this one up
                                                                                                                                # @QUESTION: Why do we need to scale up num_warmup_steps when we are using multiple GPU with accelerator?
                                     num_training_steps=Config.training_config.num_training_steps)
        self.loss_fn=torch.nn.CrossEntropyLoss()
        self.model,self.optimizer,self.train_dataloader,self.test_dataloader,self.scheduler=self.accelerator.prepare(
            self.model,self.optimizer,self.train_dataloader,self.test_dataloader,self.scheduler
        )
        self.accelerator.register_for_checkpointing(self.scheduler) # Save checkpoint is crucial since training any Tranformer-based model is very very time consuming
        
        # Resume from checkpoint
        if Config.training_config.resume_from_checkpoint is not None:
            checkpoint_dir=os.path.join(self.experiment_dir,Config.training_config.resume_from_checkpoint)
            with self.accelerator.main_process_first(): # @QUESTION: What does main_process_first() return?
                self.accelerator.load_state(checkpoint_dir) # @QUESTION: What does method load_state() expect to get in directory checkpoint_dir?
            self.completed_steps=int(checkpoint_dir.split('_')[-1])
            self.accelerator.print(f'Loaded model from checkpoint {self.completed_steps}')
        else:
            self.completed_steps=0
        self.device=self.accelerator.device
    def fit(self):
        train=True
        progress_bar=tqdm(range(self.completed_steps,self.training_config.num_training_steps),disable=not self.accelerator.is_main_process)
        while train:
            accumulate_steps=0
            accumulate_loss=0
            accuracy=0
            for batch in self.train_dataloader:
                src_input_ids=batch['src_input_ids'].to(self.device)
                src_pad_mask=batch['src_pad_mask'].to(self.device)
                tgt_input_ids=batch['tgt_input_ids'].to(self.device)
                tgt_pad_mask=batch['tgt_pad_mask'].to(self.device)
                tgt_output_ids=batch['tgt_output_ids'].to(self.device)
                
                # Predict
                outputs=self.model(src_ids=src_input_ids,
                                  tgt_ids=tgt_input_ids,
                                  src_attention_mask=src_pad_mask,
                                  tgt_attention_mask=tgt_pad_mask) # This contains probabilities over all possible next toknes

                print(outputs.shape) 
                outputs=outputs.flatten(0,1) # @QUESTION: How does flatten(0,1) work?
                tgt_output_ids=tgt_output_ids.flatten()
                loss=self.loss_fn(outputs,tgt_output_ids)
                loss=loss/self.training_config.gradient_accumulation_steps
                accumulate_loss+=loss # Since we have the above line, in the future we will not need to average this accumualte_loss anymore
                self.accelerator.backward(loss) # @QUESTION: Why are we calculating gradients on loss here? It must be calculated on accumulate_loss after gradient_accumulation_steps steps?
                preds=outputs.argmax(axis=-1)
                mask=(tgt_output_ids!=-100) # Mask to filter out padding tokens
                preds=preds[mask] # Remove all tokens with value -100
                tgt_output_ids=tgt_output_ids[mask] # Remove all tokens with value -100
                acc=(preds==tgt_output_ids).sum()/len(preds)
                accuracy+=acc/self.training_config.gradient_accumulation_steps
                accumulate_steps+=1
                
                # Update model parameters evary gradient_accumulation_steps steps
                if accumulate_steps%self.training_config.gradient_accumulation_steps==0:
                    self.accelerator.clip_grad_norm_(parameters=self.model.parameters(),max_norm=1.0) # @QUESTION: What does max_norm mean?
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True) # QUESTION: What does set_to_none mean?
                    self.scheduler.step()
                    
                    # Logging every logging_interval steps
                    if self.completed_steps%self.training_config.logging_interval==0:
                        accumulate_loss=accumulate_loss.detach() # @QUESTION: Why do we need to detach here?
                        accuracy=accuracy.detach() # @QUESTION: Why do we need to detach here?
                        
                        # @QUESTION: If more than 1 process is running, what we have to do? (explain the following code for answer)
                        if self.accelerator.num_processes>1:
                            accumulate_loss=torch.mean(self.accelerator.gather_for_metrics(accumulate_loss))
                            accuracy=torch.mean(self.accelerator.gather_for_metrics(accuracy))

                        log={
                            'train_loss':accumulate_loss,
                            'training_acc':accuracy,
                            'learning_rate':self.scheduler.get_last_lr()[0]
                        }
                        self.accelerator.log(log,step=self.completed_steps)
                        logging_string=f'[{self.completed_steps}/{self.training_config.num_training_steps}] Training Loss: {accumulate_loss} | Training Accuracy: {accuracy}'
                        if self.accelerator.is_main_process:
                            progress_bar.write(logging_string)
                    
                    # Evaluate model every evaluation_steps steps
                    if self.completed_steps%self.training_config.evaluation_steps==0:
                        self.model.eval()
                        self.accelerator.print('Evaluating model...')
                        test_losses=[]
                        test_accs=[]
                        for batch in tqdm(self.test_dataloader,disable=not self.accelerator.is_main_process):
                            src_input_ids=batch['src_input_ids'].to(self.device)
                            src_pad_mask=batch['src_pad_mask'].to(self.device)
                            tgt_input_ids=batch['tgt_input_ids'].to(self.device)
                            tgt_pad_mask=batch['tgt_pad_mask'].to(self.device)
                            tgt_output_ids=batch['tgt_output_ids'].to(self.device)
                            with torch.inference_mode(): # @QUESTION: What is the difference between torch.no_grad() vs torch.inference_mode()?
                                outputs=self.model(src_ids=src_input_ids,
                                                   tgt_ids=tgt_input_ids,
                                                   src_attention_mask=src_pad_mask,
                                                   tgt_attention_mask=tgt_pad_mask)
                                
                            # Flatten for loss calculation
                            outputs=outputs.flatten(0,1)
                            tgt_output_ids=tgt_output_ids.flatten()
                            
                            # Compute loss
                            loss=self.loss_fn(outputs,tgt_output_ids)
                            
                            # Compute accuracy
                            preds=outputs.argmax(axis=-1)
                            mask=(tgt_output_ids!=-100) # Mask to filter out padding tokens
                            preds=preds[mask] # Remove all padding tokens
                            tgt_output_ids=tgt_output_ids[mask] # Remove all padding tokens
                            acc=(preds==tgt_output_ids).sum()/len(preds)
                            
                            # Store result
                            loss=loss.detach()
                            acc=acc.detach()
                            if self.accelerator.num_processes>1:
                                loss=torch.mean(self.accelerator.gather_for_metrics(loss))
                                acc=torch.mean(self.accelerator.gather_for_metrics(acc))
                            
                            # Store metrics
                            test_losses.append(loss.item())
                            test_accs.append(acc.item())
                        test_loss=np.mean(test_losses)
                        test_acc=np.mean(test_accs)
                        log={
                            'test_loss':test_loss,
                            'test_acc':test_acc,
                        }
                        logging_string=f'Testing Loss: {test_loss} | Testing Accuracy: {test_acc}'
                        if self.accelerator.is_main_process:
                            progress_bar.write(logging_string)
                        
                        # Log and save model
                        self.accelerator.log(log,step=self.completed_steps)
                        self.accelerator.save_state(os.path.join(self.experiment_dir,f'checkpoint_{self.completed_steps}'))
                        
                        # Test traslating a sentence
                        if self.accelerator.is_main_process:
                            src_ids=src_ids[:1].to(self.device) # @STAR
                                                                # @QUESTION: Why should not we use src_ids[0] instead of src_ids[:1] here?
                            
                            # @STAR: Now since model is wrapped into accelerator, we can not access its method like model.inference() anymore, we are just be able to call method forward() via model(...), so now we need to unwrapped it for testing purpose
                            unwrapped_model=self.accelerator.unwrap_model(self.model) 
                            
                            translated=unwrapped_model.inference(src_ids=src_ids,
                                                                 tgt_start_id=self.tgt_tokenizer.special_tokens['[BOS]'],
                                                                 tgt_end_id=self.tgt_tokenizer.special_tokens['[EOS]'],
                                                                 max_len=512)    
                            translated=self.tgt_tokenizer.decode(input=translated,
                                                                 skip_special_tokens=True)
                            self.accelerator.print(translated)
                            '''if self.accelerator.is_main_process:
                                progress_bar.write(f'Translation: {translated}')'''
                        self.model.train()
                    
                    # @STAR
                    # @QUESTION: Why do we put the endpoint checking step here, in the batch loop, and after every gradient_accumulation_steps steps?
                    if self.completed_steps>self.training_config.num_training_steps:
                        self.accelerator.save_state(os.path.join(self.experiment_dir,f'final_checkpoint'))
                        train=False 
                        break
                    
                    self.completed_steps+=1
                    progress_bar.update(1)
                    accumulate_loss=0
                    accuracy=0
        self.accelerator.end_training(  )
    def get_num_parameters(self):
        num_params=0
        for param in self.model.parameters():
            if param.requires_grad:
                num_params+=param.numel() # @STAR
        return num_params
def main():
    trainer=Trainer()
    
    # Define a sample for testing purpose
    a_beautiful_sentence="Hello Everyone's Legend, I am T1!"
    src_ids=torch.tensor(trainer.src_tokenizer(a_beautiful_sentence)['input_ids']).unsqueeze(0)
    
    # Start training first
    print("Starting training...")
    trainer.fit()
    
    # Test the model after training
    print("Training completed. Testing the model...")
    trainer.model.eval()
    with torch.no_grad():
        # Move src_ids to device
        src_ids = src_ids.to(trainer.device)
        
        # Get unwrapped model for inference
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        
        # Test translation
        translated = unwrapped_model.inference(src_ids=src_ids,
                                              tgt_start_id=trainer.tgt_tokenizer.special_tokens['[BOS]'],
                                              tgt_end_id=trainer.tgt_tokenizer.special_tokens['[EOS]'],
                                              max_len=512)
        
        # Decode the translation
        translated_text = trainer.tgt_tokenizer.decode(input=translated, skip_special_tokens=True)
        
        print(f"Original English: {a_beautiful_sentence}")
        print(f"Translated French: {translated_text}")
    
    print("Training and testing completed!")
if __name__=='__main__':
    main()