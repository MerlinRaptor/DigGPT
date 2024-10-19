import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


#-------------------------

class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # getting qkv in a concatenated tensor for more computational efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # projection to mix muti-head
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_INIT = True # flag for needing residual initialization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.blocksize, config.blocksize)).view(1,1,config.blocksize,config.blocksize))
        # actually mask, following the huggingface naming

    def forward(self, x):
        B, T, C = x.size()
        # batch size, sequence leghth (1024 for GPT2), channel (embedding dimensionality)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        # distribute channels to heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)  nh:number of heads; hs: headsize
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention pattern
        att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1))) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1) 

        change = att @ v # (B, nh, T, hs)
        change = change.transpose(1, 2).contiguous().view(B, T, C) # important to use .contiguous()
        # mix muti-heads
        change = self.c_proj(change)
        return change


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # channel fully connected 
        # to study part: this dimension ascending and superposition (sparse autoencoder)
        self.gelu = nn.GELU(approximate='tanh')  # use a approximation to submit to the original version of GPT-2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_INIT = True

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# A block is a hidden layer containing a communication part(attention) and a connection part(FFn)
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # pre-layernorm, consisitent with GPT2
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # wrong code:
        # the wrong code below is not acutually, any mechanism inconsistent with the original setting would possibly lead to weight mismatch
        # x = self.ln_1(x)
        # x = x + self.attn(x)  
        # x = self.ln_2(x)
        # x = x + self.mlp(x)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    blocksize: int = 1024
    vocabsize: int = 50257
    n_layer: int = 12
    n_head:int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # the names are supposed to align with huggingface's statedict for implementing huggingface's transformer parameter values
                wte = nn.Embedding(config.vocabsize, config.n_embd), # weight of token embedding 
                wpe = nn.Embedding(config.blocksize, config.n_embd), # weight of position embedding
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layers
                ln_f = nn.LayerNorm(config.n_embd), # laynorm in the final
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocabsize, bias = False)
        # weight sharing (tying) scheme (saving roghuly 1/3 paras)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.init_weight) # applying weight initialization for all the modules

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02  # roughly consistent with Javier init
            if hasattr(module, 'RESIDUAL_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # intialization for residual path, actually 2*n_layer layers of residual layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # wpe.weight is actually intialized with std=0.01 in original GPT2

    def forward(self, idx, label=None):
        # idx: (B, T)
        B, T = idx.size()
        assert T <= self.config.blocksize, f'Sequence of length {T} cannot be longer than {self.config.blocksize}'
        pos = torch.arange(T, dtype=torch.long, device=idx.device) # T
        pos_embd = self.transformer.wpe(pos) # (T, n_embd)
        tok_embd = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_embd + pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # logits (B, T, vocabsize)
        loss = None
        if label is not None:  # (B, T)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), label.view(-1))
        return logits, loss

    @classmethod
    def get_pretrained(cls):
        model_type = 'gpt2' # 124M version
        from transformers import GPT2LMHeadModel    
        print(f'loading weights of pretrained {model_type}')

        config = GPTConfig()
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # this is a buffer rather than a para

        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # Openai trsnposed some weights for some reson
        for k in sd_keys:
            if any(k.endswith(trans) for trans in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    


#--------------------------------------------------------

import tiktoken

class Dataloader:

    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open('E:\Code\makemore\DignityGPT\input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch is {len(self.tokens) // (B * T)} batches')
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buff = self.tokens[self.current_position: self.current_position + B * T + 1]  # remember to plus one to get the last label of the last input
        x = buff[:-1].view(B, T) # input
        y = buff[1:].view(B, T)  # label
        self.current_position += B * T # advance the index with a stride of B * T
        if self.current_position + B * T + 1> len(self.tokens):
            self.current_position = 0
        return x, y

import time

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f'using device:{device}')

train_loader = Dataloader(B = 4, T = 1024) 

# torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)

# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() # synchronize gpu and cpu for timing accuracy
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokenpers = (x.shape[0] * x.shape[1]) / (t1 - t0)
    print(f'{i} step, current loss: {loss}, time for a batch: {dt}ms, token/dt: {tokenpers:.2f}')

import sys; sys.exit(0)
# generate
model.eval()
generate_batch = 5
max_length = 100

tokens = enc.encode('Hi, I want to tell you a great story: long long ago,') # T = 15 currently
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(generate_batch, 1) # (B ,T)
x = tokens.to('cuda')
while x.size(-1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1) # (5, 50)
        next_token_topkindex = torch.multinomial(topk_probs, 1) # (5, 1)
        next_token_vocabindex = torch.gather(topk_indicies, -1, next_token_topkindex)
        x = torch.cat((x, next_token_vocabindex), dim=-1)

for i in range(generate_batch):
    tokens = x[i, :max_length].tolist()
    print(f'>>>{enc.decode(tokens)}')
    # with open('initialGeneration.txt', 'a') as f:
    #     f.write(f'>>>{enc.decode(tokens)} \r\n')


# config = GPTConfig()
# model = GPT(config)
# sd = model.state_dict()
# for k,v in sd.items():
#     print(k, v.shape)
