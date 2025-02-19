import datetime
from Configs.configs import ModelConfig


def save_model_parameters(model, vocab_size, tokenization_type="Character"):
    """
    Save model parameters and configuration to a text file
    Args:
        model: The GPT model instance
        vocab_size: Size of the vocabulary
        tokenization_type: Type of tokenization used (Character/Word)
    """
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'parameters_{tokenization_type}_{timestamp}.txt'
    
    with open(filename, 'w') as f:
        # Model parameters
        f.write(f"{tokenization_type} Tokenized model parameters: {sum(p.numel() for p in model.parameters())/1e6}M parameters\n")
        
        # Configuration details
        f.write("\n=== Model Configuration ===\n")
        f.write(f"Tokenization Type: {tokenization_type} Tokenization\n")
        f.write(f"Vocabulary Size: {vocab_size}\n")
        f.write(f"Model Architecture: {model.__class__.__name__}\n\n")
        
        # Model hyperparameters
        f.write("=== Hyperparameters ===\n")
        f.write(f"Learning Rate: {ModelConfig.learning_rate}\n")
        f.write(f"Batch Size: {ModelConfig.batch_size}\n")
        f.write(f"Block Size: {ModelConfig.block_size}\n")
        f.write(f"Max Iterations: {ModelConfig.max_iters}\n")
        f.write(f"Evaluation Interval: {ModelConfig.eval_interval}\n")
        f.write(f"Evaluation Iterations: {ModelConfig.eval_iter}\n")
        f.write(f"Number of Epochs: {ModelConfig.n_epochs}\n")
        
        # Model architecture details
        f.write("\n=== Architecture Details ===\n")
        f.write(f"Device: {ModelConfig.device}\n")
        f.write(f"Dropout: {ModelConfig.dropout}\n")
        f.write(f"Number of Attention Heads: {ModelConfig.n_head}\n")
        f.write(f"Number of Layers: {ModelConfig.n_layer}\n")
        f.write(f"Embedding Dimension: {ModelConfig.n_embed}\n")