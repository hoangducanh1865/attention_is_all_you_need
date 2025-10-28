import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
class Attention_(nn.Module):
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
class SelfAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,attn_p=0.0,proj_p=0.0,bias=False):
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
    def forward(self,input,attention_mask):
        batch_size,seq_len,embed_dim=input.shape
        q=self.query(input).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k=self.key(input).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v=self.value(input).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        attention=(q @ k.transpose(-2,-1))/self.head_dim**0.5
        # Do masking
        if attention_mask is not None:
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,seq_len,1) # (3, 1, 5, 5) compare to (3, 3, 5, 5) - (batch_size, num_heads, seq_len, seq_len), head dimension will be automatically broadcast from 1 to 3
                                                                                          # We can even do not use repeat() here since every dimension except the last one will be automatically bradcast
            attention=attention.masked_fill(~attention_mask,float('-inf'))
            '''print(attention)
            print(attention.shape)'''
        attention=attention.softmax(axis=-1)
        attention=self.attn_drop(attention)
        output=attention@v
        output=output.transpose(1,2)
        output=output.reshape(batch_size,seq_len,embed_dim)
        output=self.proj(output)
        output=self.proj_drop(output)
        return output
class CausalSelfAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,causal=False,attn_p=0.0,proj_p=0.0,bias=False):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        self.causal=causal
        self.query=nn.Linear(embed_dim,embed_dim,bias=bias)
        self.key=nn.Linear(embed_dim,embed_dim,bias=bias)
        self.value=nn.Linear(embed_dim,embed_dim,bias=bias)
        self.attn_drop=nn.Dropout(attn_p)
        self.proj=nn.Linear(embed_dim,embed_dim,bias=bias) # Layer which help heads to know each others
        self.proj_drop=nn.Dropout(proj_p)
    def forward(self,input,attention_mask):
        batch_size,seq_len,embed_dim=input.shape
        q=self.query(input).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k=self.key(input).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v=self.value(input).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        attention=(q @ k.transpose(-2,-1))/self.head_dim**0.5
        if attention_mask is not None:
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,seq_len,1) # (3, 1, 5, 5) compare to (3, 3, 5, 5) - (batch_size, num_heads, seq_len, seq_len), head dimension will be automatically broadcast from 1 to 3
                                                                                            # We can even do not use repeat() here since every dimension except the last one will be automatically bradcast
        if self.causal:
            ones=torch.ones((seq_len,seq_len),device=Config.DEVICE)
            causal_mask=torch.tril(ones).bool().reshape(1,1,seq_len,seq_len) # We can use unsqueeze(0).unsqueeze(0) instead of reshape(1,1,seq_len,seq_len)
            
            # Do masking on padding tokens
            if attention_mask is not None:
                causal_mask=causal_mask.repeat(batch_size,1,1,1) # Again, we can ignore this line since torch will automatically broadcast every dimiensions except the last one
                causal_mask=causal_mask.masked_fill(~attention_mask,False)
                print(causal_mask)
                '''print(attention)
                print(attention.shape)'''
            attention=attention.masked_fill(~causal_mask,float('-inf'))
        attention=attention.softmax(axis=-1)
        attention=self.attn_drop(attention)
        output=attention@v
        output=output.transpose(1,2)
        output=output.reshape(batch_size,seq_len,embed_dim)
        output=self.proj(output)
        output=self.proj_drop(output)
        return output
class CrossAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,attn_p=0.0,proj_p=0.0,bias=False):
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
    def forward(self,src,tgt,attention_mask):
        '''print(tgt.shape)'''
        batch_size,src_seq_len,embed_dim=src.shape
        batch_size,tgt_seq_len,embed_dim=tgt.shape
        q=self.query(tgt).reshape(batch_size,tgt_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k=self.key(src).reshape(batch_size,src_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v=self.value(src).reshape(batch_size,src_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        attention=(q @ k.transpose(-2,-1))/self.head_dim**0.5
        # Do masking
        if attention_mask is not None:
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,tgt_seq_len,1) # (3, 1, 5, 5) compare to (3, 3, 5, 5) - (batch_size, num_heads, seq_len, seq_len), head dimension will be automatically broadcast from 1 to 3
                                                                                          # We can even do not use repeat() here since every dimension except the last one will be automatically bradcast
            attention=attention.masked_fill(~attention_mask,float('-inf'))
            '''print(attention)
            print(attention.shape)'''
        attention=attention.softmax(axis=-1)
        attention=self.attn_drop(attention)
        output=attention@v
        output=output.transpose(1,2)
        output=output.reshape(batch_size,tgt_seq_len,embed_dim)
        '''print(output.shape)'''
        output=self.proj(output)
        output=self.proj_drop(output)
        return output
class Attention(nn.Module):
    def __init__(self,embed_dim=768,num_heads=12,attn_drop=0.0,proj_drop=0.0):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        assert embed_dim%num_heads==0, 'Embedding dimension is not divisible by number of heads'
        self.head_dim=embed_dim//num_heads
        self.attn_drop=attn_drop
        self.proj_drop=proj_drop
        self.q_proj=nn.Linear(embed_dim,embed_dim)
        self.k_proj=nn.Linear(embed_dim,embed_dim)
        self.v_proj=nn.Linear(embed_dim,embed_dim)
        self.out_proj=nn.Linear(embed_dim,embed_dim)
        self.proj_drop=nn.Dropout(p=proj_drop)
    def forward(self,src,tgt=None,attention_mask=None,causal=False):
        batch_size,src_len,embed_dim=src.shape
        if tgt is None: # Self attention
            q=self.q_proj(src).reshape(batch_size,src_len,self.num_heads,self.head_dim).transpose(1,2)
            k=self.k_proj(src).reshape(batch_size,src_len,self.num_heads,self.head_dim).transpose(1,2)
            v=self.v_proj(src).reshape(batch_size,src_len,self.num_heads,self.head_dim).transpose(1,2)
            if attention_mask is not None:
                attention_mask=attention_mask.bool()
                attention_mask=attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,src_len,1)
            # Use flash attention via torch formula
            attention_output=F.scaled_dot_product_attention(q,k,v,
                                                            attn_mask=attention_mask,
                                                            dropout_p=self.attn_drop if self.training else 0.0,
                                                            is_causal=causal)
        else: # Cross attention
            tgt_len=tgt.shape[1]
            q=self.q_proj(tgt).reshape(batch_size,tgt_len,self.num_heads,self.head_dim).transpose(1,2)
            k=self.k_proj(src).reshape(batch_size,src_len,self.num_heads,self.head_dim).transpose(1,2)
            v=self.v_proj(src).reshape(batch_size,src_len,self.num_heads,self.head_dim).transpose(1,2)
            if attention_mask is not None:
                attention_mask=attention_mask.bool()
                attention_mask=attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,tgt_len,1)
            # Use flash attention via torch formula
            attention_output=F.scaled_dot_product_attention(q,k,v,
                                                            attn_mask=attention_mask,
                                                            dropout_p=self.attn_drop if self.training else 0.0,
                                                            is_causal=causal)
        output=attention_output.transpose(1,2).flatten(2) # flatten(2) means to flatten from dimension 2 to make (batch_size, seq_len, num_heads, head_dim) become (batch_size, seq_len, embed_dim)
        output=self.out_proj(output)
        output=self.proj_drop(output)
        return output
'''def test_00():
    rand=torch.rand(4,64,128)
    attention=Attention(embed_dim=128)
    rand=attention(rand)
    print(rand.shape)'''
'''def test_01():
    rand=torch.rand(4,64,128)
    attention=MultiHeadAttention(embed_dim=128,num_heads=4)
    rand=attention(rand)
    print(rand.shape)'''
'''def test_02():
    rand=torch.rand(4,16,128)
    attention=SelfAttentionEncoder(embed_dim=128,num_heads=2)
    rand=attention(rand)
    print(rand.shape)'''
"""def test_03():
    seq_lens=[3,5,4]
    batch_size=len(seq_lens)
    max_len=max(seq_lens)
    embed_dim=9
    num_heads=3
    attention=SelfAttention(embed_dim=embed_dim,num_heads=num_heads,causal=True)
    rand=torch.rand(batch_size,max_len,embed_dim)
    '''print(rand.shape)'''
    '''mask=torch.nn.utils.rnn.pad_sequence([torch.ones(l) for l in seq_lens],batch_first=True,padding_value=0).bool()'''
    mask=[torch.ones(l) for l in seq_lens]
    mask=torch.nn.utils.rnn.pad_sequence(mask,batch_first=True,padding_value=0) # batch_first means to merge existed tensors in to a single tensor of shape (batch_size, max_len)
    mask=mask.bool()
    output=attention(rand,mask)"""
'''def test_04():
    english_seq_lens=[3,5,4]
    french_seq_lens=[7,6,2]
    embed_dim=18
    num_heads=3
    attn=CrossAttention(embed_dim=embed_dim,num_heads=num_heads)
    rand_english=torch.randn(len(english_seq_lens),max(english_seq_lens),embed_dim)
    rand_french=torch.randn(len(french_seq_lens),max(french_seq_lens),embed_dim)
    print('Random English:')
    print(rand_english.shape)
    print('Random French:')
    print(rand_french.shape)
    english_masks=torch.nn.utils.rnn.pad_sequence([torch.ones(l) for l in english_seq_lens],batch_first=True,padding_value=0).bool()
    french_masks=torch.nn.utils.rnn.pad_sequence([torch.ones(l) for l in french_seq_lens],batch_first=True,padding_value=0).bool()
    print('English masks are:')
    print(english_masks)
    print('French masks are:')
    print(french_masks)
    output=attn(src=rand_english,tgt=rand_french,attention_mask=english_masks)'''
def test_05():
    ### Test Out Self-Attention!! ###
    print("TESTING SELF-ATTENTION!!!")
    print("-------------------------")

    seq_lens = [3, 5, 4]
    embed_dim = 9
    num_heads = 3
    a = Attention(embed_dim, num_heads)

    ### Create a random tensor in the shape (Batch x Seq Len x Embed Dim) ###
    ### This will be a tensor up to the max(seq_lens) ###
    rand = torch.randn(len(seq_lens), max(seq_lens), embed_dim)

    ### Create Attention Mask from the seq_lens (shortest sequences padded to the longest) ###
    masks = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(l) for l in seq_lens],
        batch_first=True,
        padding_value=0
    ).bool()
    print("Attention Mask:")
    print(masks)

    ### Pass through MHA ###
    output = a(rand, attention_mask=masks, causal=True)
    print("Final Output:", output.shape, "\n")
def test_06():
    print("TESTING CROSS-ATTENTION!!!")
    print("--------------------------")

    ### Test out Cross Attention
    ### We will now have sequences of different lengths, identify the number of tokens in each sequence ###
    english_seq_lens = [3, 5, 4]
    french_seq_lens = [7, 6, 2]

    embed_dim = 9
    num_heads = 3
    a = Attention(embed_dim, num_heads)

    ### Create random tensor in the shape (Batch x Seq Len x Embed Dim) for French and English ###
    ### This will be a tensor up to the max(seq_lens) ###
    rand_english = torch.randn(len(english_seq_lens), max(english_seq_lens), embed_dim)
    rand_french = torch.randn(len(french_seq_lens), max(french_seq_lens), embed_dim)

    ### Create Attention Mask from the seq_lens (shortest sequences padded to the longest) ###
    english_masks = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(l) for l in english_seq_lens],
        batch_first=True,
        padding_value=0
    ).bool()

    french_masks = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(l) for l in french_seq_lens],
        batch_first=True,
        padding_value=0
    ).bool()

    print("English Attention Mask:")
    print(english_masks)
    print("French Attention Mask:")
    print(french_masks)

    ### Pass through MHA ###
    output = a(src=rand_english, tgt=rand_french, attention_mask=english_masks)
    print("Final Output:", output.shape, "\n")
if __name__=='__main__':
    test_06()