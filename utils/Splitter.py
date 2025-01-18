import torch

class DataSplitter:
    def __init__(self, data, tokenizer, split_ratio):
        self.data = data
        self.tokenizer = tokenizer
        self.split_ratio = split_ratio
    
    def split(self, ratio):
        # Convert data to tensor using tokenizer
        encoded = self.tokenizer.encode(self.data)
        data = torch.tensor(encoded, dtype=torch.long)
        
        # Split data
        n = int(ratio * len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data, val_data

