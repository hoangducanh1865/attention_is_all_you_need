import torch
import torch.nn as nn
import torch.functional as F
class SelfAttentionEncoder(nn.Module):
    def __init__(self,embed_dim,num_heads,attn_p=0.0,proj_p=0.0,bias=False): # attn stand for attention, is dropout rate
                                                                 # proj stand for projection, is dropout rate in layer which help heads to know each others
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        self.query=nn.Linear(embed_dim,embed_dim,bias=bias)
        self.key=nn.Linear(embed_dim,embed_dim,bias=bias)
        self.value=nn.Linear(embed_dim,embed_dim,bias=bias)
        self.attn_drop=nn.Dropout(attn_p)
        self.proj=nn.Linear(embed_dim,embed_dim,bias=bias) # Layer which help heads to know each others
        self.proj_drop=nn.Dropout(proj_p)
    def forward(self,input):
        batch_size,seq_len,embed_dim=input.shape
        q=self.query(input).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k=self.key(input).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v=self.value(input).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        attention=(q @ k.transpose(-2,-1))/self.head_dim**0.5
        attention=attention.softmax(axis=-1)
        attention=self.attn_drop(attention)
        output=attention@v
        output=output.transpose(1,2)
        output=output.reshape(batch_size,seq_len,embed_dim)
        output=self.proj(output)
        output=self.proj_drop(output)
        return output
class Attention(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.embed_dim=embed_dim
        self.query=nn.Linear(embed_dim,embed_dim)
        self.key=nn.Linear(embed_dim,embed_dim)
        self.value=nn.Linear(embed_dim,embed_dim)
    def forward(self,input):
        q=self.query(input)
        k=self.key(input)
        v=self.value(input)
        similarity=(q @ k.transpose(1,2))/self.embed_dim**0.5
        attention=similarity.softmax(axis=-1) # axis=2
        output=attention@v
        return output
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        self.multihead_qkv=nn.ModuleList()
        for head in range(num_heads):
            qkv_proj=nn.ModuleDict(
                [
                    ['Q',nn.Linear(embed_dim,self.head_dim)],
                    ['K',nn.Linear(embed_dim,self.head_dim)],
                    ['V',nn.Linear(embed_dim,self.head_dim)]
                ]
            )
            self.multihead_qkv.append(qkv_proj)
        self.proj=nn.Linear(embed_dim,embed_dim)
    def forward(self,input):
        heads_out=[]
        for head in self.multihead_qkv:
            q=head['Q'](input)
            k=head['K'](input)
            v=head['V'](input)
            similarity=(q @ k.transpose(1,2))/self.head_dim**0.5
            attention=similarity.softmax(axis=-1)
            output=attention@v 
            heads_out.append(output)
        heads_out=torch.cat(heads_out,dim=-1) # dim=2, shape is (batch_size, seq_len, embed_dim)
        heads_out=self.proj(heads_out)
        return heads_out
def test_00():
    rand=torch.rand(4,64,128)
    attention=Attention(embed_dim=128)
    rand=attention(rand)
    print(rand.shape)
def test_01():
    rand=torch.rand(4,64,128)
    attention=MultiHeadAttention(embed_dim=128,num_heads=4)
    rand=attention(rand)
    print(rand.shape)
def test_02():
    rand=torch.rand(4,16,128)
    attention=SelfAttentionEncoder(embed_dim=128,num_heads=2)
    rand=attention(rand)
    print(rand.shape)
if __name__=='__main__':
    test_02()