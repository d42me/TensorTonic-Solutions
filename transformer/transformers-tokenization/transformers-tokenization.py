import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        vocab = []
        
        for text in texts:
            text_words = text.split(' ')

            for word in text_words:
                vocab.append(word)

        uniq_vocab = list(set(vocab))
        vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token] + sorted(uniq_vocab)
        self.word_to_id = {word: id for id, word in enumerate(vocab)}
        self.id_to_word = {id: word for id, word in enumerate(vocab)}

        print(self.word_to_id)
        print(self.id_to_word)
        self.vocab_size = len(vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in words]
             
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return ' '.join(self.id_to_word.get(idx, self.unk_token) for idx in ids)
