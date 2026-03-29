import torch
import torch.nn as nn
import math

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        '''
        embed_dim: The size of the mathematical vector used to represent a word (e.g., 256).
        num_heads: How many "parallel brains" you want looking at the sentence simultaneously (e.g., 8).
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        '''
        x (A tensor of shape [batch_size, seq_length, embed_dim] representing your embedded words)
        performs multi-head attention mechanism
        '''
        batch_size, seq_length, _ = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        
        mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0).unsqueeze(0).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    '''
    embed_dim: The size of the mathematical vector used to represent a word (e.g., 256).
    hidden_dim: The size of the intermediate layer in the feed-forward network (e.g., 512).
    '''
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)    
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attention = CausalMultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attended = self.attention(self.norm1(x))
        x = x + attended
        forwarded = self.ffn(self.norm2(x))
        x = x + forwarded
        return x

class MiniGPTBase(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(embed_dim, num_heads, hidden_dim))
            
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length).unsqueeze(0).to(x.device)
        
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x

