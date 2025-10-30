import os
import torch
from dataclasses import dataclass
class Config:
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR=os.path.join('data','datasets','raw_datasets','english2french')
    TRAINED_FRENCH_TOKENIZER_PATH=os.path.join('data','trained_tokenizer','french_wp.json')
    SAVE_RAW_DIR=os.path.join('data','datasets','preprocessed_datasets','english2french','hf_all_data')
    SAVE_TOK_DIR=os.path.join('data','datasets','preprocessed_datasets','english2french','hf_tokenized')
    PRETRAINED_ENGLISH_TOKENIZER_NAME='google-bert/bert-base-uncased'
@dataclass
class TransformerConfig:
    embed_dim: int=512
    num_heads: int=8 # Number of attention heads
    attn_p: float=0.0 # Attention dropout percentage
    hid_p: float=0.0 # Hidden dropout percentage
    mlp_ratio: int=4
    encoder_depth: int=6
    decoder_depth: int=6
    src_vocab_size: int=30522
    tgt_vocab_size: int=32000
    src_max_len: int=512
    tgt_max_len: int=512
    learn_pos_embed: bool=False