import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re
from Configs.configs import ModelConfig
from Model.model import GptModel
from Tokenization.tokenization import GptTokenizer
from utils.get_batch import BatchGenerator

GptTokenizer = GptTokenizer()


device = ModelConfig.device
torch.manual_seed(1337)

with open('Poems.txt','r',encoding='utf-8') as f:
    text = f.read()

    
    
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(ModelConfig.eval_iter, device=device)  # Move tensor to correct device
        for k in range(ModelConfig.eval_iter):
            x, y = BatchGenerator.get_batch(split)
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
    
    xb,yb=BatchGenerator.get_batch('train')
    
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
## save the model in directory weights
torch.save(model.state_dict(),'weights/CharaterTokenizedModel.pth')
context=torch.zeros(1,1,dtype=torch.long,device=device)
print(decode(model.generate(context,1000)[0].tolist()))