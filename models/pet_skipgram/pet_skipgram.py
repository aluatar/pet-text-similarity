import torch 
from torch import nn
from .preprocessing.tokenizer import PetTokenizer
from numpy.random import choice
import numpy as np
from typing import Literal

class PetSkipGramModel(nn.Module, PetTokenizer):
    
    def __init__(
        self, 
        vocabulary: list[str], 
        embedding_dim: int | None=50, 
        n_samples: int | None = 100, 
        temerature: float | None = 1.0,
        device: Literal['cuda' , 'cpu'] | None='cuda'
    ):
        nn.Module.__init__(self)
        PetTokenizer.__init__(self, vocabulary=vocabulary)
        self.embedding_dim = embedding_dim
        self.vocab_size = len(self.vocabulary)
        
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.out_layer = nn.Linear(in_features=self.embedding_dim, out_features=self.vocab_size)
        self.activation_function = nn.LogSoftmax(dim=1)
        self.probability_retieval = nn.Softmax(dim=1)
        
        self.n_samples = n_samples
        self.temperature = temerature
        self.word_embedding_dict = {}
        if device is not None:
            self.device = device
        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
            print(f"Model passed to CUDA device {torch.cuda.get_device_name()}")
        elif torch.cuda.is_available() and device == 'cpu':
            self.device = 'cpu'
            print(f"Device manually set to 'cpu' while CUDA device {torch.cuda.get_device_name()} is avalible. Consider switching to CUDA for better efficiency.")
        else:
            self.device = 'cpu'
            print("CUDA device is not avalible. Switching to CPU")
        
    def forward(self, center_word_idx):
        onehot_vector = torch.LongTensor([center_word_idx]).to(device=self.device)
        embeddings = self.embeddings(onehot_vector)
        logits = self.out_layer(embeddings) / self.temperature
        log_probabilities = self.activation_function(logits)
        return log_probabilities
    
    def fit(self, word) -> str:
        if word in self.word_idx_dict.keys():
            token = self.word_idx_dict[word]
        else:
            return "#oov"
        
        onehot_vector = torch.LongTensor([token])
        embeddings = self.embeddings(onehot_vector)
        logits = self.out_layer(embeddings) / self.temperature
        probabilities = self.probability_retieval(logits).data.numpy()[0]
        
        tokens = choice(range(self.vocab_size), size=self.n_samples, p=probabilities)
        tokens, counts = np.unique(tokens, return_counts=True)
        result_token = tokens[np.where(counts==max(counts))]
        return self.idx_word_dict.get(result_token[0])
        
            