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
from utils.save_params import save_model_parameters
import matplotlib.pyplot as plt
from utils.visualizer import Visualizer

device = ModelConfig.device
torch.manual_seed(1337)

class Trainer:
    def __init__(self, model, optimizer, tokenizer, train_data, val_data):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.train_data = train_data.long()
        self.val_data = val_data.long()
        self.train_losses = []
        self.val_losses = []
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=ModelConfig.max_iters,
            eta_min=1e-5
        )
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(ModelConfig.eval_iter, device=device)
            for k in range(ModelConfig.eval_iter):
                x, y = BatchGenerator(self.train_data,self.val_data).get_batch(split)
                # Ensure input tensors are in long format
                x, y = x.long().to(device), y.long().to(device)
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
    
    save_model_parameters(model,vocab_size,"Sentence Piece Tokenization")
    
    # Initialize trainer
    trainer = Trainer(model, optimizer, tokenizer, train_data, val_data)
    
    # Train for n_epochs
    for epoch in range(ModelConfig.n_epochs):
        trainer.train_epoch(epoch)
    
    # Save model and plot losses
    torch.save(model.state_dict(), 'weights/WordTokenizedModel.pth')
    vis=Visualizer("plots/O(N)_Lineformer")
    vis.plot_training_metrics(trainer.train_losses, trainer.val_losses)
    vis.plot_learning_curve(trainer.epoch_train_losses, trainer.epoch_val_losses)
    
if __name__ == "__main__":
    main()
