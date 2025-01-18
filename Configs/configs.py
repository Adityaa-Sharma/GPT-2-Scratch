from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    
    # Training parameters
    batch_size: int = 64
    block_size: int = 256
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iter: int = 200
    learning_rate: float = 6e-4 # gpt paper
    
    # Model architecture
    n_embed: int = 768
    n_layer: int = 12 # gpt-2 124M model
    n_head: int = 12 # gpt-2 124M model
    dropout: float = 0.1
    
    # System
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
