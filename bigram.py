import torch 
import torch.nn as nn
import torch.nn .functional as F
torch.manual_seed(1337)

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

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

#simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embed_tabel=nn.Embedding(vocab_size,vocab_size) # each token directly reads off the logits for the next token from a lookup table
    
    def forward(self,idx,targets=None):#id and and target are integer tensor
        logits=self.token_embed_tabel(idx)  # B,T,C  ->  (batch,time(block),channels(vocab))
     
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
            logits,loss=self(idx)   #get the prediction         
            logits=logits[:,-1,:]   # focus only in the last time stamp -> becomes( B*C )
            probs=F.softmax(logits,dim=-1) #get prob
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next), dim=1)
            #print(idx)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)  #directly move the table to the gpu


#optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


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
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))