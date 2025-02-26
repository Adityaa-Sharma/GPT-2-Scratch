import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import re
from Configs.configs import ModelConfig
from Model.model import GptModel
from TextFormatter.formatter import WordFormatter,CharacterFormatter
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
                x, y = BatchGenerator(self.train_data,self.val_data).get_batch(split)
                x, y = x.to(device), y.to(device)
                logits, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train_epoch(self, epoch):
        iters_per_epoch = ModelConfig.max_iters // ModelConfig.n_epochs
        for iter in range(iters_per_epoch):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Get batch and compute loss
            xb, yb = BatchGenerator(self.train_data, self.val_data).get_batch('train')
            logits, loss = self.model(xb, yb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            if iter % ModelConfig.eval_interval == 0:
                losses = self.estimate_loss()
                print(f'Epoch {epoch}, Iter {iter}, '
                      f'Train loss: {losses["train"]:.4f}, '
                      f'Val loss: {losses["val"]:.4f}, '
                      f'LR: {self.scheduler.get_last_lr()[0]:.2e}')
                self.train_losses.append(losses["train"])
                self.val_losses.append(losses["val"])
                self.epoch_train_losses.append(losses["train"])
                self.epoch_val_losses.append(losses["val"])
            
            xb, yb = BatchGenerator(self.train_data,self.val_data).get_batch('train')
            # Ensure input tensors are in long format
            xb, yb = xb.long().to(device), yb.long().to(device)
            logits, loss = self.model(xb, yb)
        
        # Store average losses for this epoch
        self.epoch_train_losses.append(torch.tensor(self.epoch_train_losses).mean().item())
        self.epoch_val_losses.append(torch.tensor(self.epoch_val_losses).mean().item())
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        train_losses = np.array(self.train_losses)
        val_losses = np.array(self.val_losses)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
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
    
    # Preprocess text into words
    formatter = WordFormatter(text)
    words = formatter.preprocess()  # This should return a list of words
    
    # Create vocabulary and tokenizer
    vocab = sorted(set(words))
    tokenizer = WordTokenization(vocab)
    vocab_size = len(tokenizer.word_to_idx)  # Updated to include special tokens
    
    # For WordTokenization, keep text as list of words
    splitter = DataSplitter(words, tokenizer, 0.9)
    train_data, val_data = splitter.split(0.9)
    
    #batch generator
    Batch=BatchGenerator(train_data,val_data)
    
    # Initialize model and optimizer
    model = GptModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(
    model.parameters(),
                lr=ModelConfig.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
    
    # Save model parameters info
    with open('parameters.txt', 'w') as f:
        f.write(f"Word Tokenized model parameters: {sum(p.numel() for p in model.parameters())/1e6}M parameters")
        
    #i wnat to save proper reporrt like what are the configuration , which tokenizer is used, what is the vocab size, what is the model architecture, what is the learning rate, what is the batch size, what is the block size, what is the max_iters, what is the eval_interval, what is the eval_iter, what is the n_epochs, what is the device, what is the dropout, what is the n_head, what is the n_layer, what is the n_embed
    with open('parameters.txt', 'a') as f:
        f.write(f"\n\nConfiguration: \n{ModelConfig}\n")
        f.write(f"\nTokenizer: Word Tokenization\n")
        f.write(f"\nVocab Size: {vocab_size}\n")
        f.write(f"\nModel Architecture: {model}\n")
        f.write(f"\nLearning Rate: {ModelConfig.learning_rate}\n")
        f.write(f"\nBatch Size: {ModelConfig.batch_size}\n")
        f.write(f"\nBlock Size: {ModelConfig.block_size}\n")
        f.write(f"\nMax Iters: {ModelConfig.max_iters}\n")
        f.write(f"\nEval Interval: {ModelConfig.eval_interval}\n")
        f.write(f"\nEval Iter: {ModelConfig.eval_iter}\n")
        f.write(f"\nN Epochs: {ModelConfig.n_epochs}\n")
        f.write(f"\nDevice: {ModelConfig.device}\n")
        f.write(f"\nDropout: {ModelConfig.dropout}\n")
        f.write(f"\nN Head: {ModelConfig.n_head}\n")
        f.write(f"\nN Layer: {ModelConfig.n_layer}\n")
        f.write(f"\nN Embed: {ModelConfig.n_embed}\n")
        
    
    
    # Initialize trainer
    trainer = Trainer(model, optimizer, tokenizer, train_data, val_data)
    
    # Train for n_epochs
    for epoch in range(ModelConfig.n_epochs):
        trainer.train_epoch(epoch)
    
    # Save model and plot losses
    torch.save(model.state_dict(), 'weights/WordTokenizedModel.pth')
    trainer.plot_losses()

if __name__ == "__main__":
    main()
