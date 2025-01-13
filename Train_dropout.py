import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re
from Configs.configs import ModelConfig

# Parameters


torch.manual_seed(1337)

# Load data
with open('combine_poems.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Removing numbers
text = re.sub(r'\d+', '', text)

# Character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}

encode = lambda s: [char_to_index[char] for char in s]
decode = lambda x: ''.join([index_to_char[i] for i in x])

# Train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Function to get batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - ModelConfig.block_size, (ModelConfig.batch_size,))
    x = torch.stack([data[i:i + ModelConfig.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + ModelConfig.block_size + 1] for i in ix])
    return x.to(ModelConfig.device), y.to(ModelConfig.device)  # Move batches to device

# Model components
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, ModelConfig.n_embed)
        self.position_embedding_table = nn.Embedding(ModelConfig.block_size, ModelConfig.n_embed)
        self.dropout=nn.Dropout(ModelConfig.dropout)
        self.blocks = nn.Sequential(
            Block(ModelConfig.n_embed, n_head=4),
            Block(ModelConfig.n_embed, n_head=4),
            Block(ModelConfig.n_embed, n_head=4)
        )
        self.ln_f = nn.LayerNorm(ModelConfig.n_embed)
        self.lm_head = nn.Linear(ModelConfig.n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx)  # B, T, C
        pos_embed = self.position_embedding_table(torch.arange(T, device=idx.device))  # T, C
        x = tok_embed + pos_embed  # B, T, C
        x = self.blocks(x)
        logits = self.lm_head(x)  # B, T, vocab_size

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx = idx[:, -ModelConfig.block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Only get the last time step
            probs = F.softmax(logits, dim=-1)  # B, C
            idx_next = torch.multinomial(probs, 1)  # B, 1
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(ModelConfig.n_embed, head_size, bias=False)
        self.query = nn.Linear(ModelConfig.n_embed, head_size, bias=False)
        self.value = nn.Linear(ModelConfig.n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(ModelConfig.block_size, ModelConfig.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) / C**0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        y = wei @ v  # (B, T, C)
        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(ModelConfig.n_embed, ModelConfig.n_embed)
        self.dropout = nn.Dropout(ModelConfig.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embed // n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(ModelConfig.eval_iter, device=ModelConfig.device)
        for k in range(ModelConfig.eval_iter):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Dropout(ModelConfig.dropout),
            nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model and optimizer
model = Bigram(vocab_size).to(ModelConfig.device)
optimizer = optim.Adam(model.parameters(), lr=ModelConfig.learning_rate)

# Training loop
for iter in range(ModelConfig.max_iters):
    if iter % ModelConfig.eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter}, Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text
torch.save(model.state_dict(),'model_dropout.pth')
