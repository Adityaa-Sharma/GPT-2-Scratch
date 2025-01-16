import torch
from torch import nn
from Model.MultiHeadAttention import MultiHeadAttention
from Model.FFNN import FeedForward


class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size=n_embed//n_head
        self.sa=MultiHeadAttention(n_head,n_embed//n_head)
        self.ffwd=FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x=x+self.sa(x)
        x=x+self.ffwd(x)
        return x