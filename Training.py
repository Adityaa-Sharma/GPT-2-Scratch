import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re
from Configs.configs import ModelConfig
from Model.model import GptModel
from Tokenization.tokenization import GptTokenizer, CharacterTokenization, WordTokenization
from utils.get_batch import BatchGenerator
from utils.Splitter import DataSplitter
import matplotlib.pyplot as plt

GptTokenizer = GptTokenizer()

device = ModelConfig.device
torch.manual_seed(1337)

class Trainer:
    def __init__(self, model, optimizer, tokenizer, train_data, val_data):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.train_losses = []
        self.val_losses = []
        
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(ModelConfig.eval_iter, device=device)
            for k in range(ModelConfig.eval_iter):
                x, y = BatchGenerator.get_batch(split)
                x, y = x.to(device), y.to(device)
                logits, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train_epoch(self, epoch):
        iters_per_epoch = ModelConfig.max_iters // ModelConfig.n_epochs
        for iter in range(iters_per_epoch):
            if iter % ModelConfig.eval_interval == 0:
                losses = self.estimate_loss()
                print(f'Epoch {epoch}, Iter {iter}, Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}')
                self.train_losses.append(losses["train"])
                self.val_losses.append(losses["val"])
            
            xb, yb = BatchGenerator.get_batch('train')
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig('training_plot.png')
        plt.close()

def main():
    # Load and preprocess data
    with open('dataset/Poems.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = CharacterTokenization(sorted(set(text)))
    vocab_size = len(tokenizer.char_to_idx)
    
    # Split data
    splitter = DataSplitter(text, CharacterTokenization, 0.9)
    train_data, val_data = splitter.split(0.9)
    
    # Initialize model and optimizer
    model = GptModel(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ModelConfig.learning_rate)
    
    # Save model parameters info
    with open('parameters.txt', 'w') as f:
        f.write(f"{sum(p.numel() for p in model.parameters())/1e6}M parameters")
    
    # Initialize trainer
    trainer = Trainer(model, optimizer, tokenizer, train_data, val_data)
    
    # Train for n_epochs
    for epoch in range(ModelConfig.n_epochs):
        trainer.train_epoch(epoch)
    
    # Save model and plot losses
    torch.save(model.state_dict(), 'weights/CharaterTokenizedModel.pth')
    trainer.plot_losses()

if __name__ == "__main__":
    main()