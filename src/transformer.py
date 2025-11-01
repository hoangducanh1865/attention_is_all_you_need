import torch
import torch.nn as nn
import torch.nn.functional as F 
from src.config import Config
from src.embedding import Embeddings
from src.encoder import Encoder
from src.decoder import Decoder
from src.utils import Utils
class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.encodings=Embeddings(config)
        self.encoder=nn.ModuleList(
            [Encoder(config) for _ in range(config.encoder_depth)]
        )
        self.decoder=nn.ModuleList(
            [Decoder(config) for _ in range(config.decoder_depth)]
        )
        self.head=nn.Linear(config.embed_dim,config.tgt_vocab_size)
        self.apply(Utils._init_weights_) # Apply this function to modify how torch initialize layers in your own way
    def forward(self,src_ids,tgt_ids,src_attention_mask=None,tgt_attention_mask=None):
        src_embeddings=self.encodings(input_ids=src_ids,type='source')
        tgt_embeddings=self.encodings(input_ids=tgt_ids,type='target')
        
        # Loop over all encoder blocks
        for layer in self.encoder:
            src_embeddings=layer(x=src_embeddings,attention_mask=src_attention_mask)
        
        # Loop over all decoder blocks
        for layer in self.decoder:
            tgt_embeddings=layer(src=src_embeddings,
                                 tgt=tgt_embeddings,
                                 src_mask=src_attention_mask,
                                 tgt_mask=tgt_attention_mask)
        preds=self.head(tgt_embeddings)
        return preds
    @torch.no_grad()
    def inference(self,src_ids,tgt_start_id=2,tgt_end_id=3,max_len=512):
        tgt_ids=torch.tensor([tgt_start_id],device=src_ids.device).reshape(1,1) # (batch_size, seq_len) currently is (1, 1)
        src_embeddings=self.encodings(input_ids=src_ids,type='source')
        for layer in self.encoder:
            src_embeddings=layer(x=src_embeddings) # @QUESTION: Why do we set attention_mask=None here when batch_size is just 1?
        for i in range(max_len-1):
            tgt_embeddings=self.encodings(input_ids=tgt_ids,type='target')
            
            # Loop over all decoder blocks
            for layer in self.decoder:
                tgt_embeddings=layer(src=src_embeddings,
                                     tgt=tgt_embeddings) # @QUESTION: Again, why with batch_size = 1, we just set both src_mask and tgt_mask equal to None here?
            tgt_embeddings=tgt_embeddings[:,-1] # For each batch, we just need to take the last embedding, since this already contains the context of tokens from the first one to this one
            preds=self.head(tgt_embeddings)
            preds=preds.argmax(axis=1).unsqueeze(0)
            tgt_ids=torch.cat([tgt_ids,preds],dim=-1) # Concatenate along sequence dimension
            print(tgt_ids)
            if torch.all(preds==tgt_end_id):
                break 
        return tgt_ids.squeeze().cpu().tolist() # Squeeze to remove batch_size dimension since we do not need it anymore
                                                # @QUESTION: How does squeeze work here?
                                                # @ANSWER: It removes all dimensions of size 1, if we just simply need to remove only dimension 0 (if its size is 1) then we should use squeeze(0)
def main():
    config=Config.transformer_config
    t=Transformer(config)
    eng=torch.randint(low=0,high=1000,size=(1,32))
    fre=torch.randint(low=0,high=1000,size=(1,43))
    t.inference(eng)
if __name__=='__main__':
    main()