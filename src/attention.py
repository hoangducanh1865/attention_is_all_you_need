import torch
import torch.nn as nn
import torch.functional as F
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
            similarity=(q @ k.transpose(1,2))/self.embed_dim**2
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
if __name__=='__main__':
    test_01()