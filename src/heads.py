import torch
import torch.nn as nn
from model import MiniGPTBase

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len):
        super().__init__()
        self.backbone = MiniGPTBase(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.lm_head(features)
        return logits
        
    def generate(self, start_tokens, max_new_tokens, max_seq_len):
        self.eval()
        current_tokens = start_tokens
        
        for i in range(max_new_tokens):
            seq_len = current_tokens.size(1)
            
            if seq_len > max_seq_len:
                context = current_tokens[:, -max_seq_len:]
            else:
                context = current_tokens
                
            logits = self(context)
            last_token_logits = logits[:, -1, :]
            
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            current_tokens = torch.cat((current_tokens, next_token), dim=1)
            
        return current_tokens

class GPTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len, num_classes=3):
        super().__init__()
        self.backbone = MiniGPTBase(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len)
        self.classifier_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        last_token_features = features[:, -1, :]
        logits = self.classifier_head(last_token_features)
        return logits