import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re

batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iter=200
n_embed=32

torch.manual_seed(1337)

with open('combine_poems.txt','r',encoding='utf-8') as f:
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
    ix=torch.randint(0,len(data)-block_size,(batch_size,))
    # print("printing ix: ", ix)

    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

class Bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__() ## call the parent class constructor
        self.token_embedding_table=nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table=nn.Embedding(block_size,n_embed)
        self.sa_head=Head(n_embed)
        self.lm_head=nn.Linear(n_embed,vocab_size)  
        
        
    def forward(self, idx,targets=None):
        B,T=idx.shape
        tok_embed=self.token_embedding_table(idx) # B T C
        pos_embed=self.position_embedding_table(torch.arange(T,device=idx.device)) # T C
        x=tok_embed+pos_embed # B T C
        x=self.sa_head(x)
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
            idx=idx[:,-block_size:]
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
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size))) # registering trill as it was not a parameter
        
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

    
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter, device=device)  # Move tensor to correct device
        for k in range(eval_iter):
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
        self.net=nn.Sequential(nn.Linear(n_embed,n_embed),nn.ReLU())
        
        
    def forward(self,x):
        return self.net(x)



model = Bigram(vocab_size)

optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if iter%eval_interval==0:
        losses=estimate_loss()
        print(f'Iter {iter}, Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}')
    
    xb,yb=get_batch('train')
    
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context=torch.zeros(1,1,dtype=torch.long)
print(decode(model.generate(context,1000)[0].tolist()))