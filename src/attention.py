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
def test():
    rand=torch.rand(4,64,128)
    attention=Attention(embed_dim=128)
    rand=attention(rand)
    print(rand.shape)
if __name__=='__main__':
    test()