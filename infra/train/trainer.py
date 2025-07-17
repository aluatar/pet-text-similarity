import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from models.dataset import Dataset
from models.pet_skipgram.pet_skipgram import PetSkipGramModel

class PetSkipGramTrainer:
    def __init__(self, 
        model: PetSkipGramModel, 
        loss: _Loss,
        training_data: Dataset, 
        validation_data: Dataset | None = None, 
        batch_size: int | None = 1,
        learning_rate: float | None = 2e-4,
        shuffle: bool | None = True,
        epoches: int | None = 1,
        optimizer: Optimizer | None = torch.optim.SGD
    ):
        self.model = model
        self.loss = loss()
        self.training_data = training_data
        self.validation_data = validation_data
        self.batch_size = batch_size 
        self.shuffle = shuffle 
        self.epoches = epoches
        self.device = self.model.device
              
        self.train_dataloader = DataLoader(
            dataset=training_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle, 
            
        )
        if validation_data is not None:
            self.validation_dataloader = DataLoader(
                dataset=validation_data,
                batch_size=self.batch_size,
                shuffle=self.shuffle
            )
            
        self.optimizer = optimizer(params=self.model.parameters(), lr=learning_rate)
        
        self.convergece = {}
            
    
    def train(self):
        self.model.to(self.device)
        self.train_dataloader
        for epoch in range(self.epoches):
            loss = 0
            for center_word, context_word in tqdm(self.train_dataloader):
                self.model.zero_grad()
                log_probabilities = self.model.forward(center_word)
                _loss = self.loss(log_probabilities.cpu(), context_word)
                _loss.backward()
                self.optimizer.step()
                
                loss += _loss.item()
            
            print(f"Epoch: {epoch + 1}, Loss: {loss}")
            self.convergece[epoch] = loss
        embeddings = self.model.embeddings.weight.data.numpy()
        self.model.words_embedding_dict = {
            self.model.idx_word_dict[idx]: embeddings[idx] 
            for idx in range(self.model.vocab_size)
        }
            
    
    