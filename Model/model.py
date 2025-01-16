import torch
from Configs.configs import ModelConfig
import torch.nn as nn
import torch.nn.functional as F
from Model.Block import Block

class GptModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__() ## call the parent class constructor
        self.token_embedding_table=nn.Embedding(vocab_size,ModelConfig.n_embed)
        self.position_embedding_table=nn.Embedding(ModelConfig.block_size,ModelConfig.n_embed)
        self.blocks=nn.Sequential(
            Block(ModelConfig.n_embed,n_head=4),
            Block(ModelConfig.n_embed,n_head=4),
            Block(ModelConfig.n_embed,n_head=4))  # 4 heads
        # self.ffwd=FeedForward(n_embed)
        self.ln_f = nn.LayerNorm(ModelConfig.n_embed)
        self.lm_head=nn.Linear(ModelConfig.n_embed,vocab_size)  
        
        
    def forward(self, idx,targets=None):
        B,T=idx.shape
        tok_embed=self.token_embedding_table(idx) # B T C
        pos_embed=self.position_embedding_table(torch.arange(T,device=idx.device)) # T C
        x=tok_embed+pos_embed # B T C
        x=self.blocks(x)
        logits=self.lm_head(x) # B T Vocab_size
        # print("logits", logits)
        
        if targets is None:
            loss=None
        else:
            
            # print("shape of logits: ", logits.shape)
            # but pytorch expects B C T
            B,T,C=logits.shape
        
            logits=logits.view(B*T,C)
            # print("shape of logits after view: ", logits.shape)
            # print("shape of targets: ", targets.shape)
            targets=targets.view(B*T)
            # print("shape of targets: ", targets.shape)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
        

    def generate(self,idx,max_new_token):
        for _ in range(max_new_token):
            idx=idx[:,-ModelConfig.block_size:]
            logits,loss=self(idx)
            logits=logits[:,-1,:] #only getting the (B,C)
            probs=F.softmax(logits,dim=-1) # B,C
            idx_next=torch.multinomial(probs,1) # B,1
            # print("idx: ", idx)
            idx=torch.cat([idx,idx_next],dim=1)
            
        return idx
