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
    embedding_dimension: int=512
    num_attention_heads: int=8
    attention_droput: float=0.0
    hidden_dropout: float=0.0
    mlp_ratio: int=4
    encoder_depth: int=6
    decoder_depth: int=6
    src_vocab_size: int=30522
    tgt_vocab_size: int=32000
    max_src_len: int=512
    max_tgt_len: int=512
    learn_pos_embed: bool=False