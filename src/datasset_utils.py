import pandas as pd
import torch
from torch.utils.data import Dataset

def load_and_split_corpus(file_path, train_ratio=0.9):
    df = pd.read_csv(file_path)
    text_data = df.iloc[:, 0].dropna().tolist()
    
    split_idx = int(len(text_data) * train_ratio)
    train_text = text_data[:split_idx]
    val_text = text_data[split_idx:]
    
    return train_text, val_text

class LanguageModelingDataset(Dataset):
    def __init__(self, data_tokens, seq_length):
        self.data_tokens = data_tokens
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data_tokens) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.data_tokens[idx : idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y