import torch
import os
from datasets import load_dataset,concatenate_datasets,load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.tokenizer import FrenchTokenizer
from src.config import Config
class DataManager:
    def __init__(self,datasets_dir,save_raw_dir,save_tok_dir,vocab_path,english_tokenizer_name):
        self.datasets_dir=datasets_dir
        self.save_raw_dir=save_raw_dir
        self.save_tok_dir=save_tok_dir
        self.vocab_path=vocab_path
        self.english_tokenizer_name=english_tokenizer_name
    def build_english2french_dataset(self,test_p=0.005):
        hf_datasets=[]
        for dir in os.listdir(self.datasets_dir):
            dataset_dir=os.path.join(self.datasets_dir,dir)
            if os.path.isdir(dataset_dir):
                print(f'Processing dataset: {dataset_dir}')
                french_text_file=None
                english_text_file=None
                for txt_file in os.listdir(dataset_dir):
                    if txt_file.endswith('.fr'):
                        french_text_file=os.path.join(dataset_dir,txt_file)
                    elif txt_file.endswith('.en'):
                        english_text_file=os.path.join(dataset_dir,txt_file)
                if french_text_file is not None and english_text_file is not None:
                    french_text=load_dataset('text',data_files=french_text_file)['train']
                    english_text=load_dataset('text',data_files=english_text_file)['train']
                    print(french_text)
                    print(english_text)
                    english_text=english_text.rename_column('text','english_src')
                    dataset=english_text.add_column('french_tgt',french_text['text'])
                    hf_datasets.append(dataset)
        hf_datasets=concatenate_datasets(hf_datasets)
        print(hf_datasets)
        hf_dataset=hf_datasets.train_test_split(test_size=test_p) # We can use method train_test_split() in here because method concatenate_datasets() on hf_datasets - which is a list of class Dataset objects returned from method load_dataset() - return a Dataset class object, which has method train_test_split()
        hf_dataset.save_to_disk(self.save_raw_dir)
    def tokenize_english2french_dataset(self,num_workers=2,truncate=True,max_length=512,min_length=5):
        french_tokenizer=FrenchTokenizer(vocab_path=self.vocab_path,
                                         truncate=truncate,
                                         max_length=max_length)  
        english_tokenizer=AutoTokenizer.from_pretrained(self.english_tokenizer_name)
        raw_dataset=load_from_disk(self.save_raw_dir)
        def tokenize_text(examples):
            english_text=examples['english_src']
            french_text=examples['french_tgt']
            src_ids=english_tokenizer(english_text,truncation=truncate,max_length=max_length)['input_ids']
            tgt_ids=french_tokenizer.encode(input=french_text)
            
            # We have to return a dictionary here since method map() of library datasets (Hugging Face datasets) demand for it
            batch={
                'src_ids':src_ids,
                'tgt_ids':tgt_ids
            }
            
            return batch
        tokenized_dataset=raw_dataset.map(tokenize_text,batched=True,num_proc=num_workers)
        tokenized_dataset=tokenized_dataset.remove_columns(['english_src','french_tgt']) # Remove these columns since we do not need them any more
        filter_fn=lambda examples: [len(e)>min_length for e in examples['tgt_ids']] # Len of each tokenized example in examples must longer than min_length
        tokenized_dataset=tokenized_dataset.filter(filter_fn,batched=True)
        print(tokenized_dataset)
        tokenized_dataset.save_to_disk(self.save_tok_dir)
    def prepare_padded_and_masked_english2french_dataset(self):
        dataset=load_from_disk(self.save_tok_dir)
        src_tokenizer=AutoTokenizer.from_pretrained(self.english_tokenizer_name)
        tgt_tokenizer=FrenchTokenizer(vocab_path=self.vocab_path)
        collate_fn=self.translation_collator(src_tokenizer,tgt_tokenizer)
        data_loader=DataLoader(dataset=dataset['train'],batch_size=128,collate_fn=collate_fn,shuffle=True,num_workers=4)
        for samples in tqdm(data_loader):
            pass
        return data_loader
    def translation_collator(self,src_tokenizer,tgt_tokenizer):
        def _collate_fn(batch):
            src_ids=[torch.tensor(i['src_ids']) for i in batch]
            tgt_ids=[torch.tensor(i['tgt_ids']) for i in batch]
            src_pad_token=src_tokenizer.pad_token_id # pad_token_id is an attribute of AutoTokenizer
            src_padded_ids=torch.nn.utils.rnn.pad_sequence(src_ids,batch_first=True,padding_value=src_pad_token)
            src_pad_mask=(src_padded_ids!=src_pad_token)
            tgt_pad_token=tgt_tokenizer.special_tokens['[PAD]']
            tgt_padded_ids=torch.nn.utils.rnn.pad_sequence(tgt_ids,batch_first=True,padding_value=tgt_pad_token)
            
            # @QUESTION: why do we need to use clone() here?
            # @ANSWER: We need to use clone() here since PyTorch creates a view that shares the same memory as the original tensor -> both new tensors point to the same data in memory -> during backpropagation, gradients might accumulate incorrectly; and if we modify one tensor, if will affect the other
            input_tgt_padded_ids=tgt_padded_ids[:,:-1].clone() # First dimension is batch_size, in second dimension
            output_tgt_padded_ids=tgt_padded_ids[:,1:].clone()
            output_tgt_padded_ids[output_tgt_padded_ids==tgt_pad_token]=-100 # Assign -100 for padding token so that when calculating the CrossEntropyLoss, any sample which has label (use the word "label" here since this output_tgt_padded_ids should be the output of the model) -100 will be automatically ignored
            input_tgt_pad_mask=(input_tgt_padded_ids!=tgt_pad_token)
            '''output_tgt_padded_ids=(output_tgt_padded_ids!=tgt_pad_token)''' # We do not need this padding mask
            batch={
                'src_input_ids':src_padded_ids,
                'src_pad_mask':src_pad_mask,
                'tgt_input_ids':input_tgt_padded_ids,
                'tgt_pad_mask':input_tgt_pad_mask,
                'tgt_output_ids':output_tgt_padded_ids
            }
            return batch
        return _collate_fn
def main():
    datasets_dir=Config.DATA_DIR
    save_raw_dir=Config.SAVE_RAW_DIR
    save_tok_dir=Config.SAVE_TOK_DIR
    vocab_path=Config.TRAINED_FRENCH_TOKENIZER_PATH
    english_tokenizer_name=Config.PRETRAINED_ENGLISH_TOKENIZER_NAME
    data_manager=DataManager(datasets_dir,save_raw_dir,save_tok_dir,vocab_path,english_tokenizer_name)
    '''data_manager.build_english2french_dataset()'''
    data_manager.tokenize_english2french_dataset(num_workers=4,
                                                 truncate=True,
                                                 max_length=512,
                                                 min_length=5)
    '''train_data_loader=data_manager.prepare_padded_and_masked_english2french_dataset()'''
if __name__=='__main__':
    main()