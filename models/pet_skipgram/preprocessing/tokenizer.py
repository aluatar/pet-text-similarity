import torch
import tokenizers
from abc import ABCMeta
from numba import jit
from numpy import unique

class PetTokenizer:
    def __init__(self, vocabulary: list[str]):
        self.vocabulary = set([])
        self.word_idx_dict = {}
        self.idx_word_dict = {}
        self.tokenize(vocabulary)
        
    def tokenize(self, vocabulary: str | list[str]) -> tuple[dict[str, int], dict[int, str]]:
        current_vocabulary_size = len(self.vocabulary)
        vocabulary = list(unique(vocabulary))
        word_idx_pairs = [
            (str(vocabulary[i - current_vocabulary_size]), i)
            for i in range(current_vocabulary_size, len(vocabulary) + current_vocabulary_size)
            if vocabulary[i - current_vocabulary_size] not in self.vocabulary
        ]
        
        for word, idx in word_idx_pairs:
            self.word_idx_dict[word] = idx
            self.idx_word_dict[idx] = word
            
        self.vocabulary = self.vocabulary.union(set(vocabulary))
        
        
    def encode(self, text: list[str]) -> list[int]:
        _encoded = [
            self.word_idx_dict[word] if word in self.vocabulary else -1
            for word in text
        ]
        return _encoded
    
    
    def decode(self, sequence: list[int]) -> list[str]:
        _decoded = [
            self.idx_word_dict[token] if token in self.idx_word_dict.keys() else "_"
            for token in sequence
        ]
        return _decoded
        