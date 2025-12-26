import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
from src.config import TransformerConfig
class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        assert config.embed_dim%config.num_heads==0, 'Embedding dimension is not divisible by number of heads'
        self.head_dim=config.embed_dim//config.num_heads
        self.q_proj=nn.Linear(config.embed_dim,config.embed_dim)
        self.k_proj=nn.Linear(config.embed_dim,config.embed_dim)
        self.v_proj=nn.Linear(config.embed_dim,config.embed_dim)
        self.out_proj=nn.Linear(config.embed_dim,config.embed_dim)
    def forward(self,src,tgt=None,attention_mask=None,causal=False):
        batch_size,src_len,embed_dim=src.shape
        if tgt is None: # Self attention
            q=self.q_proj(src).reshape(batch_size,src_len,self.config.num_heads,self.head_dim).transpose(1,2)
            k=self.k_proj(src).reshape(batch_size,src_len,self.config.num_heads,self.head_dim).transpose(1,2)
            v=self.v_proj(src).reshape(batch_size,src_len,self.config.num_heads,self.head_dim).transpose(1,2)
            if attention_mask is not None:
                attention_mask=attention_mask.bool()
                attention_mask=attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,src_len,1)

        else: # Cross attention
            tgt_len=tgt.shape[1]
            q=self.q_proj(tgt).reshape(batch_size,tgt_len,self.config.num_heads,self.head_dim).transpose(1,2)
            k=self.k_proj(src).reshape(batch_size,src_len,self.config.num_heads,self.head_dim).transpose(1,2)
            v=self.v_proj(src).reshape(batch_size,src_len,self.config.num_heads,self.head_dim).transpose(1,2)
            if attention_mask is not None:
                attention_mask=attention_mask.bool()
                attention_mask=attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,tgt_len,1)
        # Use flash attention via torch formula
        attention_output=F.scaled_dot_product_attention(q,k,v,
                                                        attn_mask=attention_mask,
                                                        dropout_p=self.config.attn_drop if self.training else 0.0,
                                                        is_causal=causal)
        output=attention_output.transpose(1,2).flatten(2) # flatten(2) means to flatten from dimension 2 to make (batch_size, seq_len, num_heads, head_dim) become (batch_size, seq_len, embed_dim)
        output=self.out_proj(output)
        return output
def main():
    config=Config.transformer_config
    attn=Attention(config)
    rand_eng=torch.rand(2,35,512)
    rand_fre=torch.rand(2,63,512)
    print(attn(src=rand_eng,tgt=rand_fre).shape)
if __name__=='__main__':
    main()
