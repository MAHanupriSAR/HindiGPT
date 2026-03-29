import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_utils import load_classification_data, TextClassificationDataset, load_and_split_corpus
from tokeniser import HindiTokenizer
from heads import GPTClassifier

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == y).sum().item()
        total += y.size(0)
        
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
            
    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def run_classification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = 5000
    embed_dim = 256
    num_heads = 8
    hidden_dim = 512
    num_layers = 4
    max_seq_len = 128
    batch_size = 32
    epochs = 5
    num_classes = 3
    
    train_text, _ = load_and_split_corpus("data/raw/hindi_corpus/train")
    tokenizer = HindiTokenizer()
    tokenizer.train(train_text, vocab_size=vocab_size)
    
    train_df, val_df = load_classification_data("data/raw/text_classification_dataset/train.csv")
    
    train_dataset = TextClassificationDataset(train_df, tokenizer, max_seq_len)
    val_dataset = TextClassificationDataset(val_df, tokenizer, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = GPTClassifier(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_seq_len, num_classes).to(device)
    
    pretrained_dict = torch.load("gpt_language_model.pth", map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.startswith("backbone")}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "gpt_classifier.pth")

if __name__ == "__main__":
    run_classification()