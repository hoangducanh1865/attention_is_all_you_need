import torch
import torch.nn as nn
import torch.nn.functional as F 
from src.attention import Attention
from src.feed_forward import FeedForward
class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.enc_attn=Attention(config)
        self.dropout=nn.Dropout(config.hid_drop)
        self.layer_norm=nn.LayerNorm(config.embed_dim) # @QUESTION: Why do we use LayerNorm instead of BatchNorm? What about other kinds of normalization like GroupNorm, etc.?
                                                       # @QUESTION: What is the difference between these types of normalization?
        self.feed_forward=FeedForward(config)
        self.final_layer_norm=nn.LayerNorm(config.embed_dim)
    def forward(self,x,attention_mask=None): 
        # Residual connection
        x=x+self.dropout(self.enc_attn(src=x,attention_mask=attention_mask,causal=False)) # @QUESTION: Why do we keep casual=False here?
        x=self.layer_norm(x)
        
        # Another residual connection
        x=x+self.feed_forward(x)
        x=self.final_layer_norm(x)
        
        return x