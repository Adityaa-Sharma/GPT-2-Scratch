import re
import tiktoken
import sentencepiece as spm
import os


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
    
    

class SentencePieceTokenizer:
    def __init__(self, vocab_size=32000):
        self._vocab_size = vocab_size
        self.sp_model = spm.SentencePieceProcessor()
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.models_dir, exist_ok=True)

    @property
    def vocab_size(self):
        return self._vocab_size

    def train(self, input_file):
        # Create a temporary file for model prefix
        model_prefix = "tokenizer_model"
        
        # Train the tokenizer
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=self._vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            max_sentence_length=2048
        )
        
        # Load the trained model
        self.sp_model.load(f"{model_prefix}.model")
        return self

    def encode(self, text):
        return self.sp_model.encode_as_ids(text)

    def decode(self, tokens):
        return self.sp_model.decode_ids(tokens)
