import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
class PositionalEncoding(nn.Module):
    def __init__(self,max_len,embed_dim,require_grads=False):
        super().__init__()
        self.max_len=max_len
        self.embed_dim=embed_dim
        self.require_grads=require_grads
        self.encodings=self._build_positional_encodings()
    def _build_positional_encodings(self):
        encodings=torch.zeros(self.max_len,self.embed_dim)
        position_idx=torch.arange(0,self.max_len,dtype=torch.float32).reshape(-1,1) # Reshape to convert this to column matrix
        embed_skip_dim=torch.arange(0,self.embed_dim,step=2,dtype=torch.float32)
        
        # @QUESTION: why did original paper use this formula to represent the positional information between tokens?
        # @QUESTION: why use 10000 here? why not an another number?
        encodings[:,0::2]=torch.sin(position_idx/10000**(embed_skip_dim/self.embed_dim)) # encodings[:, ...] means all rows -> apply to every position
                                                                                         # 0::2 means start:stop:step -> start at index 0, go to the end, and jump by 2 -> all even columns
        encodings[:,1::2]=torch.cos(position_idx/10000**(embed_skip_dim/self.embed_dim)) # encodings[:, ...] means all rows -> apply to every position
                                                                                         # 1::2 means start:stop:step -> start at index 1, go to the end, and jump by 2 -> all odd columns
        """
        Or we can say, for every position:
            - put sine values into even dimensions
            - put cosine values into odd dimensions
        """
        
        encodings=nn.Parameter(encodings,requires_grad=self.require_grads)
        return encodings
    def forward(self,x):
        seq_len=x.shape[1] # Since x.shape is (batch_size, seq_len, embed_dim)
        encodings=self.encodings[:seq_len]
        
        # Add positional information into original data
        x=x+encodings # Remember, batch_size dimension here will be automatically broadcast
        
        return x
class Embeddings(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        # Matrices to convert tokens to vectors
        # @QUESTION: how does this nn.Embedding work?
        self.src_embeddings=nn.Embedding(num_embeddings=config.src_vocab_size,embedding_dim=config.embed_dim)
        self.tgt_embeddings=nn.Embedding(num_embeddings=config.tgt_vocab_size,embedding_dim=config.embed_dim)
        
        # Positional embeddings
        self.src_positional_embeddings=PositionalEncoding(max_len=config.src_max_len,
                                                          embed_dim=config.embed_dim,
                                                          require_grads=config.learn_pos_embed)
        self.tgt_positional_embeddings=PositionalEncoding(max_len=config.tgt_max_len,
                                                          embed_dim=config.embed_dim,
                                                          require_grads=config.learn_pos_embed)
    def forward(self,input_ids,type):
        if type=='source':
            embeddings=self.src_embeddings(input_ids)
            embeddings=self.src_positional_embeddings(embeddings)
        elif type=='target':
            embeddings=self.tgt_embeddings(input_ids)
            embeddings=self.tgt_positional_embeddings(embeddings)
        else:
            raise TypeError('type must be "source" or "target"')
        return embeddings
def main():
    pe=PositionalEncoding(max_len=512,embed_dim=512)
    print(pe)
if __name__=='__main__':
    main()
