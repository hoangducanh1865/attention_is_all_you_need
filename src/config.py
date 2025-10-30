import os
import torch
class Config:
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR=os.path.join('data','datasets','raw_datasets','english2french')
    TRAINED_FRENCH_TOKENIZER_PATH=os.path.join('data','trained_tokenizer','french_wp.json')
    SAVE_RAW_DIR=os.path.join('data','datasets','preprocessed_datasets','english2french','hf_all_data')
    SAVE_TOK_DIR=os.path.join('data','datasets','preprocessed_datasets','english2french','hf_tokenized')
    PRETRAINED_ENGLISH_TOKENIZER_NAME='google-bert/bert-base-uncased'
