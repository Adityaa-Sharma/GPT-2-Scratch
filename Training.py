import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re
from Configs.configs import ModelConfig
from Model.model import GptModel
from Tokenization.tokenization import GptTokenizer

GptTokenizer = GptTokenizer()


device = ModelConfig.device
torch.manual_seed(1337)

with open('Poems.txt','r',encoding='utf-8') as f:
    text = f.read()







# trainn and test splitting
data=torch.tensor(GptTokenizer.encode(text),dtype=torch.long)
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





model = GptModel(vocab_size).to(device)

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