import re

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
    
    def preprocess(self):

        preprocessed = re.findall(r'\w+', self.text)
        # to remove -->  h', '0o', '0w', '100th', '105th', '10th', '11th', '12th', '137th', '13th', '14th', '15th' etc
        preprocessed = [
            word for word in preprocessed 
            if len(word) > 1 and not re.search(r'\d', word) and not re.match(r'\d+(st|nd|rd|th|s)$', word.lower())
        ]
        
    
        preprocessed = sorted(set(preprocessed))
        return preprocessed
