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
    learning_rate: float = 3e-4
    
    # Model architecture
    n_embed: int = 384
    n_layer: int = 6
    n_head: int = 6
    dropout: float = 0.2
    
    # System
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
