import pandas as pd
import torch
from torch.utils.data import Dataset
from tokeniser import HindiTokenizer

#Task 1.1
def load_and_split_corpus(folder_path, train_ratio=0.9):
    '''
    input: file_path: A string pointing to your raw data (e.g., "data/raw/train.csv")
    output: A tuple containing two lists: (train_text, val_text), lists of Hindi strings
    '''
    text_data = []
    
    # Get a list of all .txt files in the folder
    search_path = os.path.join(folder_path, "*.txt")
    file_list = glob.glob(search_path)
    
    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:  # Only add if the file isn't empty
                text_data.append(content)
    
    split_idx = int(len(text_data) * train_ratio)
    train_text = text_data[:split_idx]
    val_text = text_data[split_idx:]
    
    return train_text, val_text

class LanguageModelingDataset(Dataset):
    def __init__(self, data_tokens, seq_length):
        '''
        #data_tokens: A massive, single 1D list of integer IDs (the output of tokenizer.encode() run on your entire text).
        #seq_length: An integer dictating how many tokens the model is allowed to look at in one go (e.g., 128 or 256).
        '''
        self.data_tokens = data_tokens
        self.seq_length = seq_length

    def __len__(self):
        '''
        What it does: PyTorch needs to know exactly how many valid sequences it can extract from your data. If your dataset has 1,000,000 tokens and your seq_length is 100, you cannot start a 100-token sequence at token number 999,995 (you would run out of bounds). So, it calculates the total length minus the sequence length.
        '''
        return len(self.data_tokens) - self.seq_length

    def __getitem__(self, idx):
        '''
        [10, 20, 30, 40, 50, 60]
        Let's say your sequence length (seq_length) is 3.
        PyTorch's DataLoader asks your dataset for the very first sample: my_dataset[0]. 
        Here is exactly what happens inside the function:

        Step A: Grab the Chunk -> chunk = self.data_tokens[idx : idx + self.seq_length + 1]
        The index is 0.
        We want a chunk starting at 0 and ending at 0 + 3 + 1 = 4.
        chunk grabs the first 4 tokens: [10, 20, 30, 40]
        Why grab 4 if the sequence length is 3? Because we need an extra token to act as the "future" word for the final prediction.

        Step B: Create Input ($x$) -> x = torch.tensor(chunk[:-1], dtype=torch.long)

        Step C: Create Target ($y$) -> y = torch.tensor(chunk[1:], dtype=torch.long)

        Why do we do this? 
        Put $x$ and $y$ right next to each other:
        Input ($x$): [10, 20, 30], Target ($y$): [20, 30, 40]
        Notice how every item in $y$ is exactly one step into the future compared to $x$. 
        This trains the autoregressive GPT model to predict the next word at every single step:
        Model looks at 10, it is forced to predict 20.Model looks at 10, 20, it is forced to predict 30.Model looks at 10, 20, 30, it is forced to predict 40.
        '''
        chunk = self.data_tokens[idx : idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y     


#Task 1.3
def load_classification_data(file_path, train_ratio=0.8):
    df = pd.read_csv(file_path)
    
    total_rows = len(df)
    train_size = int(total_rows * train_ratio)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    return train_df, val_df


class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        '''
        Input: The dataframe (from the function above), your trained BPE tokenizer, and a max_length (the maximum number of words/tokens allowed per review).

        What it does: It stores these tools inside the class so they are ready whenever the model asks for a piece of data.

        Output: An initialized dataset object.
        '''
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Input: idx (An integer representing which specific row to get).

        What it does:Extraction: It grabs the Hindi text and the "experience" label (0, 1, or 2) from that row. 
        Tokenization: It turns the Hindi string into a list of numbers using the encode function.
        Length Correction: * Truncation: If the review is too long (like the "Airlift" review), it cuts off the end so it fits.
        Padding: If the review is too short, it adds "empty" tokens (ID 1) to fill the space.Tensor Conversion: 
        It converts the list of numbers into a format (Tensors) that the computer's graphics card (GPU) can process.
        
        Output: A pair (x, y) where:x: A list of token IDs representing the movie review.y: A single number representing the sentiment category (0, 1, or 2).
        '''
        row = self.data.iloc[idx]
        text = str(row["text"])
        label = int(row["experience"])
        
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            padding = [1] * (self.max_length - len(tokens))
            tokens = tokens + padding
            
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y