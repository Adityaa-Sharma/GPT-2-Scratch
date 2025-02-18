import re
from collections import Counter

class CharacterFormatter:
    def __init__(self, text):
        self.text = text
    
    def preprocess(self):
        # Remove numbers
        text = re.sub(r'\d+', '', self.text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[α-ωΑ-Ω]', '', text)
        preprocessed=sorted(list(set(text)))
        return preprocessed
    

class WordFormatter:
    def __init__(self, text):
        self.text = text
        self.min_word_length = 2
        self.min_word_freq = 5  # Minimum frequency for a word to be included
        self.max_vocab_size = 50000  # Maximum vocabulary size
    
    def preprocess(self):
        # Basic word extraction
        words = re.findall(r'\w+', self.text.lower())  # Convert to lowercase
        
        # Filter words
        words = [
            word for word in words 
            if (len(word) >= self.min_word_length and  # Remove very short words
                not re.search(r'\d', word) and  # Remove words with numbers
                not re.match(r'\d+(st|nd|rd|th|s)$', word) and  # Remove ordinals
                not re.match(r'^[^a-z]+$', word))  # Remove non-letter words
        ]
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Filter by frequency and limit vocabulary size
        common_words = [
            word for word, count in word_counts.most_common(self.max_vocab_size)
            if count >= self.min_word_freq
        ]
        
        return sorted(set(common_words))
    
    def get_vocab_stats(self):
        vocab = self.preprocess()
        return {
            'vocab_size': len(vocab),
            'word_length_avg': round(sum(len(w) for w in vocab) / len(vocab), 2),
            'sample_words': vocab[:10]
        }
