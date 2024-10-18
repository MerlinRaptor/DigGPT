import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters:
bacth_size = 64 #B
block_size = 256  #T
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200  # avoid the jitter of the loss by cal the mean among lots of batches
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(42)

with open('E:\Code\makemore\DignityGPT\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
#mapping
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#data splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (bacth_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + 1 + block_size] for i in ix]) # concatenate all the blocks in a batch and add a batch dimension
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train' , 'val']:
        losses = torch.zeros(eval_iters) ### always remember to set loss to be zero
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()  ##### Remeber not to refer to the method object but the return result object of calling the mathod
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B T hs
        q = self.query(x) # B T hs
        # affinities
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]** -0.5)  #B T T
        #!!! It have a deep and vital reason to use matmul with transpose!!!
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) 
        v = self.value(x) # B T hs
        out = wei @ v # B T C
        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self ,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))
    

# a FF layer to enable the model to "think and reflect on" the connection learned previously in attention pattern, avoiding information flow to the result too quickly
class FeedFowardLayer(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFowardLayer(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Add residual connections
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLM(nn.Module):
    
    def __init__(self):  # typos here : init -> int 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.lm_head = nn.Linear(n_embd,vocab_size)
        self.positin_embedding_table = nn.Embedding(block_size, n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, 4),
        #     Block(n_embd, 4),
        #     Block(n_embd, 4),
        #     Block(n_embd, 4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B , T = idx.shape

        tok_emb = self.token_embedding_table(idx) #B, T, C
        pos_emb = self.positin_embedding_table(torch.arange(T, device=device)) #T C
        x = tok_emb + pos_emb # B T C
        x = self.blocks(x)
        x = self.ln_f(x)    
        logits = self.lm_head(x) # B T vocab_size

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, -1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #idx is B T
            idx_lastblock = idx[:, -block_size:]
            logits, loss = self(idx_lastblock) # B T vocabsize
            logits = logits[: , -1 , :] # B vocabsize only focus on the last token
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples= 1) #B 1
            idx = torch.cat((idx, idx_next), dim = -1) # cat will keep the ndim while stack will add a dim, cat is like fusin internally
        return idx
    
model = BigramLM()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step{iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # avoid ambiguity here between " and '

    xb, yb = get_batch('train')

    logits, loss = m(xb, targets = yb)
    optimizer.zero_grad(set_to_none=True) #### vital!
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
# print(decode(m.generate(context, 500)[0].tolist()))
open('output.txt','w').write(decode(m.generate(context, 10000)[0].tolist()))
