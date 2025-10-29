import os
import glob
import argparse
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFC,Lowercase # NFC to handle with unicode
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing
from src.config import Config
class MyTokenizer:
    def __init__(self,data_dir,trained_tokenizer_path):
        self.special_tokens={
            'unknown_token':'[UNK]',
            'pad_token':'[PAD]',
            'start_token':'[BOS]',
            'end_token':'[EOS]'
        }
        self.data_dir=data_dir
        self.trained_tokenizer_path=trained_tokenizer_path
    def train_tokenizer(self):
        tokenizer=Tokenizer(WordPiece(unk_token='[UNK]')) # In this project we use WordPiece, but for some popular models like GPT, Llama,... they use another one
        tokenizer.normalizer=normalizers.Sequence([NFC(),Lowercase()]) # NFC to handle with unicode (e.g: fránces), LowerCase to lower everything
        tokenizer.pre_tokenizer=Whitespace() # In this machine translation problem, both English and French have space so we use WhiteSpace() as pre_tokenizer here
        french_files=glob.glob(os.path.join(self.data_dir,'**/*.fr')) # Loop over folders and then loop over files to take files ending with .fr
        trainer=WordPieceTrainer(vocab_size=32000,
                                      special_tokens=list(self.special_tokens.values()))
        tokenizer.train(files=french_files,trainer=trainer)
        tokenizer.save(self.trained_tokenizer_path)
class FrenchTokenizer:
    def __init__(self,vocab_path,truncate=True,max_length=512): # truncate is used for deviding long words/sentences
        self.vocab_path=vocab_path
        self.truncate=truncate
        self.tokenizer=self.prepare_tokenizer()
        self.vocab_size=len(self.tokenizer.get_vocab())
        self.special_tokens={
            '[UNK]':self.tokenizer.token_to_id('[UNK]'),
            '[PAD]':self.tokenizer.token_to_id('[PAD]'),
            '[BOS]':self.tokenizer.token_to_id('[BOS]'),
            '[EOS]':self.tokenizer.token_to_id('[EOS]')
        }
        
        # Since text are going to have some tokens, and we want to add token [BOS] and token [EOS] at the biginning and the end of the text respectively
        # We can do this job manually or use TemplateProcessing like this:
        self.post_processor=TemplateProcessing(
            single='[BOS] $A [EOS]',
            special_tokens=[
                ('[BOS]',self.special_tokens['[BOS]']),
                ('[EOS]',self.special_tokens['[EOS]'])
            ]
        )
        if truncate:
            self.max_len=max_length-self.post_processor.num_special_tokens_to_add(is_pair=False) # is_pair is set to False since we are just going to processing one bt one senctence, not multiple sentences so we just need to add special tokens one time
    def prepare_tokenizer(self):
        tokenizer=Tokenizer.from_file(self.vocab_path)
        tokenizer.decoder=decoders.WordPiece() # token2word
        return tokenizer
    def encode(self,input):
        def process_tokens(tokenized):
            if self.truncate:
                tokenized.truncate(self.max_len,direction='right') 
            tokenized=self.post_processor.process(encoding=tokenized)
            return tokenized.ids
        if isinstance(input,str):
            tokenized=self.tokenizer.encode(input)
            tokenized=process_tokens(tokenized)
        if isinstance(input,list):
            tokenized=self.tokenizer.encode_batch(input)
            tokenized=[process_tokens(t) for t in tokenized]
        return tokenized
    def decode(self,input,skip_special_tokens=True):
        if isinstance(input,list):
            if all(isinstance(item,list) for item in input):
                decoded=self.tokenizer.decode_batch(input,skip_special_tokens=skip_special_tokens)
            elif all(isinstance(item,int) for item in input):
                decoded=self.tokenizer.decode(input,skip_special_tokens=skip_special_tokens)
        return decoded
def main():
    data_dir=Config.DATA_DIR
    trained_tokenizer_path=Config.TRAINED_FRENCH_TOKENIZER_PATH
    tknz=MyTokenizer(data_dir,trained_tokenizer_path)
    
    # Check if parent folders are empty or not, then check if thay have already existed or not 
    if os.path.dirname(os.path.dirname(Config.TRAINED_FRENCH_TOKENIZER_PATH)): 
        os.makedirs(os.path.dirname(Config.TRAINED_FRENCH_TOKENIZER_PATH),exist_ok=True)
    
    # Check if the trained French tokenizer has already created or not
    if not os.path.exists(Config.TRAINED_FRENCH_TOKENIZER_PATH):
        tknz.train_tokenizer()
    
    f_tknz=FrenchTokenizer(vocab_path=Config.TRAINED_FRENCH_TOKENIZER_PATH)
    sentences=['Hello World!','How are you?']
    enc=f_tknz.encode(sentences)
    print(enc)
    dec=f_tknz.decode(enc,skip_special_tokens=False)
    print(dec)
if __name__=='__main__':
    main()