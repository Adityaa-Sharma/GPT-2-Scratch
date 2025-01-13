import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re
from Configs.configs import ModelConfig


device = ModelConfig.device
torch.manual_seed(1337)

with open('Poems.txt','r',encoding='utf-8') as f:
    text = f.read()
#removing numbers
text = re.sub(r'\d+', '', text)

chars=sorted(list(set(text)))
vocab_size=len(chars)

# mapping from character to index and index to character
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

encode=lambda s:[char_to_index[char] for char in s]
decode=lambda x:''.join([index_to_char[i] for i in x])

# trainn and test splitting
data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]


def get_batch(split):
    data=train_data if split== 'train' else val_data
    ix=torch.randint(0,len(data)-ModelConfig.block_size,(ModelConfig.batch_size,))
    # print("printing ix: ", ix)

    x=torch.stack([data[i:i+ModelConfig.block_size] for i in ix])
    y=torch.stack([data[i+1:i+ModelConfig.block_size+1] for i in ix])
    x,y=x.to(ModelConfig.device),y.to(ModelConfig.device)
    return x,y

class Bigram(nn.Module):
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
    
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(ModelConfig.eval_iter, device=device)  # Move tensor to correct device
        for k in range(ModelConfig.eval_iter):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)  # Uncomment and use device
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed)
        )
            
    def forward(self,x):
        return self.net(x)



model = Bigram(vocab_size).to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer=optim.Adam(model.parameters(),lr=ModelConfig.learning_rate)

for iter in range(ModelConfig.max_iters):
    if iter%ModelConfig.eval_interval==0:
        losses=estimate_loss()
        print(f'Iter {iter}, Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}')
    
    xb,yb=get_batch('train')
    
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
## save the model
torch.save(model.state_dict(),'model.pth')
context=torch.zeros(1,1,dtype=torch.long,device=device)
print(decode(model.generate(context,1000)[0].tolist()))