import torch
import torch.nn as nn
import torch.nn.functional as F 
from src.config import TransformerConfig
class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.hid_dim=config.embed_dim*config.mlp_ratio # In original paper "Attention is All You Need", hid_size is equal to 2048 (4 * 512)
        self.intermediate_dense=nn.Linear(config.embed_dim,self.hid_dim) # @QUESTION: what does this layer do?
        self.activation=nn.GELU()
        self.intermediate_drop=nn.Dropout(config.hid_drop)
        self.output_dense=nn.Linear(self.hid_dim,config.embed_dim)
        self.output_drop=nn.Dropout(config.hid_drop)
    def forward(self,x):
        x=self.intermediate_dense(x)
        x=self.activation(x)
        x=self.intermediate_drop(x)
        x=self.output_dense(x)
        x=self.output_drop(x)
        return x
def main():
    config=TransformerConfig()
    ff=FeedForward(config)
    rand_data=torch.rand(2,32,512)
    ff(rand_data)
if __name__=='__main__':
    main()