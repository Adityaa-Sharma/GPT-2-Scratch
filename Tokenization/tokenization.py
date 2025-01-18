import re
import tiktoken

class CharacterTokenization:
    def __init__(self, vocab: list):
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    def encode(self, s: str) -> list:
        return [self.char_to_idx[char] for char in s]

    def decode(self, x: list) -> str:
        return ''.join([self.idx_to_char[i] for i in x])

class WordTokenization:
    def __init__(self, vocab):
        # Add special tokens
        vocab = ['<UNK>', '<PAD>'] + [word for word in vocab if word.strip()]
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        self.UNK_TOKEN = 0  # Index of <UNK>

    def encode(self, text: str) -> list:
        if isinstance(text, list):
            words = text
        else:
            words = text.split()
        return [self.word_to_idx.get(word, self.UNK_TOKEN) for word in words]

    def decode(self, tokens: list) -> str:
        return ' '.join([self.idx_to_word[i] for i in tokens])

class GptTokenizer:
    def __init__(self):
        self.tokenizer=tiktoken.get_encoding('gpt2')
        
    def encode(self, s: str) -> list:
        return self.tokenizer.encode(s)
    
    def decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens)