import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re

batch_size = 32
block_size = 128
max_iters = 8000
eval_interval = 500
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iter = 200
n_embed = 32
dropout = 0.2

torch.manual_seed(1337)

# Train a SentencePiece model on the input text
text_file = "combine_poems.txt"
sp_model_prefix = "sentencepiece_model"

# Train SentencePiece (only needs to be done once)
spm.SentencePieceTrainer.train(
    input=text_file, model_prefix=sp_model_prefix, vocab_size=8000, model_type='bpe'
)

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load(f"{sp_model_prefix}.model")

# Load and preprocess the text
with open(text_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Remove numbers for simplicity
text = re.sub(r'\d+', '', text)

# Encode text using SentencePiece
data = torch.tensor(sp.encode(text, out_type=int), dtype=torch.long)

# Train-test split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Helper functions for batching
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# Define the model
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4)
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx)  # B T C
        pos_embed = self.position_embedding_table(torch.arange(T, device=idx.device))  # T C
        x = tok_embed + pos_embed  # B T C
        x = self.blocks(x)
        logits = self.lm_head(x)  # B T Vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx = idx[:, -block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Only consider the last time step
            probs = F.softmax(logits, dim=-1)  # B, C
            idx_next = torch.multinomial(probs, 1)  # B, 1
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# Additional components for the transformer
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) / C**0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        y = wei @ v
        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

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

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter, device=device)
        for k in range(eval_iter):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize the model and optimizer
vocab_size = sp.get_piece_size()
model = Bigram(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter}, Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Text generation
sentence = "I love the way"
encoded_sentence = torch.tensor([sp.encode(sentence, out_type=int)], dtype=torch.long, device=device)
output = model.generate(encoded_sentence, max_new_tokens=100)
decoded_output = sp.decode(output[0].tolist())
print(decoded_output)