import torch
from Configs.configs import ModelConfig

class DataLoader:
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(0, len(data)-ModelConfig.block_size, (ModelConfig.batch_size,))
        # print("printing ix: ", ix)
        
        x = torch.stack([data[i:i+ModelConfig.block_size] for i in ix])
        y = torch.stack([data[i+1:i+ModelConfig.block_size+1] for i in ix])
        x, y = x.to(ModelConfig.device), y.to(ModelConfig.device)
        return x, y