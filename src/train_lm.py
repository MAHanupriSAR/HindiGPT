import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

from dataset_utils import load_and_split_corpus, LanguageModelingDataset
from tokeniser import HindiTokenizer
from heads import GPTLanguageModel

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            
            loss = criterion(logits, y)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def generate_hindi_text(model, tokenizer, start_text, max_new_tokens, max_seq_len, device):
    start_tokens = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long).to(device)
    
    generated_tokens = model.generate(start_tokens, max_new_tokens=max_new_tokens, max_seq_len=max_seq_len)
    
    final_text = tokenizer.decode(generated_tokens[0].tolist())
    
    return final_text

def run_language_modeling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = 5000
    embed_dim = 256
    num_heads = 8
    hidden_dim = 512
    num_layers = 4
    max_seq_len = 128
    batch_size = 32
    epochs = 5
    
    print("Loading and splitting corpus...")
    # train_text, val_text = load_and_split_corpus("data/raw/hindi_corpus/train")
    train_text, val_text = load_and_split_corpus("data/raw/hindi_corpus/merged_train.txt")
    
    print("Training tokenizer...")
    tokenizer = HindiTokenizer()
    tokenizer.train(train_text, vocab_size=vocab_size)
    
    print("Encoding tokens...")
    train_tokens = tokenizer.encode(" ".join(train_text))
    val_tokens = tokenizer.encode(" ".join(val_text))
    
    print("Creating datasets...")
    train_dataset = LanguageModelingDataset(train_tokens, max_seq_len)
    val_dataset = LanguageModelingDataset(val_tokens, max_seq_len)
    
    print("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("Initializing model...")
    model = GPTLanguageModel(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len).to(device)
    
    print("Initializing loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, perplexity = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {perplexity:.4f}")

    torch.save(model.state_dict(), "gpt_language_model.pth")

    start_prompt = "यह फिल्म"
    generated_output = generate_hindi_text(
        model=model, 
        tokenizer=tokenizer, 
        start_text=start_prompt, 
        max_new_tokens=50, 
        max_seq_len=max_seq_len, 
        device=device
    )
    
    print("\nGenerated Text:\n", generated_output)
    
    with open("generated_output.txt", "w", encoding="utf-8") as f:
        f.write(generated_output)
    

if __name__ == "__main__":
    print("Starting language modeling...")
    run_language_modeling()
