import torch
import torch.nn as nn
import torch.nn.functional as F 
from src.attention import Attention
from src.feed_forward import FeedForward
class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        
        # Self attention block 
        self.dec_attn=Attention(config)
        self.dec_attn_drop=nn.Dropout(config.hid_drop)
        self.dec_attn_layer_norm=nn.LayerNorm(config.embed_dim)
        
        # Cross attention block
        self.cross_attn=Attention(config)
        self.cross_attn_drop=nn.Dropout(config.hid_drop)
        self.cross_attn_layer_norm=nn.LayerNorm(config.embed_dim)
        
        # Feed forward block
        self.feed_forward=FeedForward(config)
        self.final_layer_norm=nn.LayerNorm(config.embed_dim)
    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
        # Self attention
        tgt=tgt+self.dec_attn_drop(self.dec_attn(src=tgt,attention_mask=tgt_mask,causal=True)) # Use causal mask here since in decoder we just be able to see past information
        tgt=self.dec_attn_layer_norm(tgt)
        
        # Cross attention
        tgt=tgt+self.cross_attn_drop(self.cross_attn(src=src,tgt=tgt,attention_mask=src_mask)) # @QUESTION: Why do we use src_mask instead of tgt_mask here?
        tgt=self.cross_attn_layer_norm(tgt)
        
        # Feed forward
        tgt=tgt+self.feed_forward(tgt)
        tgt=self.final_layer_norm(tgt)
        
        return tgt
        