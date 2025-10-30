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
        encodings[:,0::2]=torch.sin(position_idx/10000**(embed_skip_dim/self.embed_dim))
        encodings[:,1::2]=torch.cos(position_idx/10000**(embed_skip_dim/self.embed_dim))
        
        encodings=nn.Parameter(encodings,requires_grad=self.require_grads)
        return encodings
    def forward(self,x):
        seq_len=x.shape[1] # Since x.shape is (batch_size, seq_len, embed_dim)
        encodings=self.encodings[:seq_len]
        
        # Add positional information into original data
        x=x+encodings # Remember, batch_size dimension here will be automatically broadcast
        
        return x
def main():
    pe=PositionalEncoding(max_len=512,embed_dim=512)
    print(pe)
if __name__=='__main__':
    main()