from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

#task1.2
class HindiTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, text_list, vocab_size=5000): 
        #input: text_list: A list of strings (this will be the train_text you generated in Task 1.1
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
        )
        self.tokenizer.train_from_iterator(text_list, trainer=trainer)
        #output: None

    def encode(self, text): 
        #iput: A single string of Hindi text for eg "यह एक किताब है"
        encoded = self.tokenizer.encode(text)
        return encoded.ids
        #A list of integers representing the token IDs (e.g., [45, 1203, 340, 89])

    def decode(self, token_ids): 
        #input: A list of integers representing the token IDs (e.g., [45, 1203, 340, 89])
        return self.tokenizer.decode(token_ids)
        #output: A string of Hindi text (e.g., "यह एक किताब है")

    def save(self, file_path): #input: The path where you want to save the tokenizer (e.g., "hindi_tokenizer.json")
        self.tokenizer.save(file_path)
        #output: None

    def load(self, file_path): #input: The path where you want to load the tokenizer (e.g., "hindi_tokenizer.json")
        self.tokenizer = Tokenizer.from_file(file_path)
        #output: None