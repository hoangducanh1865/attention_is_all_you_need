import os
import torch
from dataclasses import dataclass
@dataclass
class TransformerConfig:
    embed_dim: int=512
    num_heads: int=8 # Number of attention heads
    attn_drop: float=0.0 # Attention dropout percentage
    hid_drop: float=0.0 # Hidden dropout percentage
    mlp_ratio: int=4
    encoder_depth: int=6
    decoder_depth: int=6
    src_vocab_size: int=30522
    tgt_vocab_size: int=32000
    src_max_len: int=512
    tgt_max_len: int=512
    learn_pos_embed: bool=False
class TrainingConfig:
    ### Dataloader Config ###
    batch_size = 128
    gradient_accumulation_steps = 2
    num_workers = 4

    ### Training Config ###
    learning_rate = 1e-4
    num_training_steps = 150000
    num_warmup_steps = 2000
    scheduler_type = "cosine"
    evaluation_steps = 250
    '''bias_norm_weight_decay = False'''
    weight_decay = 0.001
    betas = (0.9, 0.98)
    adam_eps = 1e-6

    ### Logging Config ###
    working_dir = os.path.join('data','work_dir') # Local directory
    # working_dir = '/' + os.path.join('kaggle','input','attention_is_all_you_need','data','work_dir') # Kaggle directory
    experiment_name = "seq2eq_neural_machine_translation"
    logging_interval = 1

    ### Resume from checkpoint ###
    resume_from_checkpoint = None

class Config:
    transformer_config=TransformerConfig()
    training_config=TrainingConfig()
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    
    # Local directories
    DATA_DIR=os.path.join('data','datasets','raw_datasets','english2french')
    TRAINED_FRENCH_TOKENIZER_PATH=os.path.join('data','trained_tokenizer','french_wp.json')
    SAVE_RAW_DIR=os.path.join('data','datasets','preprocessed_datasets','english2french','hf_all_data')
    SAVE_TOK_DIR=os.path.join('data','datasets','preprocessed_datasets','english2french','hf_tokenized')
    
    # Kaggle directories
    # DATA_DIR='/'+os.path.join('kaggle','input','attention_is_all_you_need','data','datasets','raw_datasets','english2french')
    # TRAINED_FRENCH_TOKENIZER_PATH='/'+os.path.join('kaggle','input','attention_is_all_you_need','data','trained_tokenizer','french_wp.json')
    # SAVE_RAW_DIR='/'+os.path.join('kaggle','input','attention_is_all_you_need','data','datasets','preprocessed_datasets','english2french','hf_all_data')
    # SAVE_TOK_DIR='/'+os.path.join('kaggle','input','attention_is_all_you_need','data','datasets','preprocessed_datasets','english2french','hf_tokenized')
    
    PRETRAINED_ENGLISH_TOKENIZER_NAME='google-bert/bert-base-uncased'