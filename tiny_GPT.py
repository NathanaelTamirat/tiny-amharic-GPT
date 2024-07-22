import torch 
import torch.nn as nn
import torch.nn .functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.manual_seed(1337)

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed=512
n_head=8
n_layer=8
dropout=0.1

with open('final_cleaned_amaharic_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(device)
#unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)
# mapping from chars to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 4 randint from 0-(len(data)-block_size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

""" one head of self-attention """
""" talk eachother(nodes)"""
class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout) #regualrize it to overcome the overfitting

    def forward(self,x):
        # input of size (batch, time-step, channels)
        B,T,C=x.shape
        k=self.key(x)     # (B,T,hs)
        q=self.query(x) # (B,T,hs)
        v=self.value(x) # (B,T,hs)
 
        wei=q@k.transpose(-2,-1) * k.shape[-1]**0.5# scaling   # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        '''  decoder block '''
        '''  cant compute the future'''
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # -> (B, T, T)
        wei=F.softmax(wei,dim=-1) # -> (B, T, T)
        out=wei@v  # (B,T,T) @ (B,T,hs)------>(B,T,hs)
        # output of size (batch, time-step, head size)
        return out

""" multpile head of self attention(in parallel)"""
class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(head_size*num_heads,n_embed)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        out= torch.cat([h(x) for h in self.heads],dim=-1)
        out= self.dropout(self.proj(out))
        return out
    
"""think on the data individually(each token)"""
class FeedForward(nn.Module):
    
    def __init__(self,n_embed):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.GELU(),   # nn.ReLU(),
            nn.Linear(4* n_embed,n_embed),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
    
""" transformer block"""
class Block(nn.Module):

    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size=n_embed//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embed)
        self.ln1=nn.LayerNorm(n_embed) #normalize the layer
        self.ln2=nn.LayerNorm(n_embed)

    
    """ residual connection is to add the original input (or a modified version 
    of it) to the output of a deeper layer. This helps mitigate the degradation 
    of gradient information as it flows backward through multiple layers during 
    training. Residual connections enable the network to learn incremental changes
    rather than trying to learn the entire transformation from scratch.
    """

    def forward(self,x):
        x=x+self.sa(self.ln1(x)) #normalize the layer before self attention
        x=x+self.ffwd((self.ln2(x)))  #normalize the layer before self attention 
        return x


'''simple bigram model'''
class GPTLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embed_table=nn.Embedding(vocab_size,n_embed) # each token directly reads off the logits for the next token from a lookup table
        self.position_embedding_table=nn.Embedding(block_size,n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])  # 6 layer of block
        self.ln_f=nn.LayerNorm(n_embed) # the final layer norm
        self.lm_head=nn.Linear(n_embed,vocab_size) # (32,65)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.2)      

    def forward(self,idx,targets=None):
        B,T=idx.shape
        #id and and target are integer tensor
        token_embed=self.token_embed_table(idx)  # B,T,n_embed  ->  (batch,time(block),n_embed)
        post_embed=self.position_embedding_table(torch.arange(T,device=device))
        x = self.ln_f(token_embed + post_embed) #x=token_embed+post_embed  # Added layer norm after embedding 
        x=self.blocks(x)
        logits=self.lm_head(x)  # B,T,vocab_size 

        if targets is None:
            loss=None
        else:
            #pytorch arrange it differnt way and the dimension
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)    
        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:] #from behind of the T upto -8 position
            logits,loss=self(idx_cond)   #get the prediction         
            logits=logits[:,-1,:]   # focus only in the last time stamp -> becomes( B*C )
            probs=F.softmax(logits,dim=-1) #get prob
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next), dim=1)
        return idx

model = GPTLM()
m = model.to(device)  #directly move the table to the gpu
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# #optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=max_iters)


for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
   
   # loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('final.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

with open('final.txt', 'w', encoding='utf-8') as file:
    file.write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))