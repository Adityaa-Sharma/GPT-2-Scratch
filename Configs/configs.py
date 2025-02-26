from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    # Training parameters
    batch_size: int = 8 
    block_size: int = 256 
    max_iters: int = 480000
    eval_interval: int = 5000  
    eval_iter: int = 50
    learning_rate: float = 3e-4  
    n_epochs: int = 2
    
    # Model architecture
    n_embed: int = 384  
    n_layer: int = 6   
    n_head: int = 6    
    dropout: float = 0.1
    
    # Learning rate schedule
    warmup_steps: int = 2000
    lr_decay: bool = True
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
