import re

#character tokenizer
class CharacterTokenization:
    def __init__(self, vocab: list):
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    

    def tokenize(self, s: str) -> list:
        encode=lambda s:[self.char_to_idx[char] for char in s]
        return encode(s)

    def detokenize(self, x: list) -> str:
        decode=lambda x:''.join([self.idx_to_char[i] for i in x])
        return decode(x)
    
    

class WordTokenization:
    def __init__(self, vocab):  # Remove list type hint for now
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocab)}

    def tokenize(self, s: str) -> list:
        encode=lambda s:[self.word_to_idx[word] for word in s.split()]
        return encode(s)

    def detokenize(self, tokens: list) -> str:
        decode=lambda tokens: ' '.join([self.idx_to_word[i] for i in tokens])
        return decode(tokens)

