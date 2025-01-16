import torch
from Configs.configs import ModelConfig
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(ModelConfig.n_embed,head_size,bias=False)
        self.query=nn.Linear(ModelConfig.n_embed,head_size,bias=False)
        self.value=nn.Linear(ModelConfig.n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(ModelConfig.block_size,ModelConfig.block_size))) # registering trill as it was not a parameter
        
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        
        wei=q@k.transpose(-2,-1)/C**0.5 # (B,T,C) @ (B,C,T) = (B,T,T)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        
        y=wei@v # (B,T,T) @ (B,T,C) = (B,T,C)
        return y
    
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(ModelConfig.n_embed, ModelConfig.n_embed)
        self.dropout = nn.Dropout(ModelConfig.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out