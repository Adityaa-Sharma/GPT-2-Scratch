
import torch
class DataSplitter:
    def __init__(self, data, tokenizer, ratio):
        self.data = data
        self.tokenizer = tokenizer()
        self.ratio = ratio
        

    def split(self, ratio):
        data=torch.tensor(self.tokenizer.encode(self.data),dtype=torch.long)
        n=int(self.ratio*len(data))
        train_data=data[:n]
        val_data=data[n:]
        return train_data, val_data
        
