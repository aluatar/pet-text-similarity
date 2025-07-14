import torch 
from torch.utils.data import Dataset
from infra.train.texts import Text
from pathlib import Path

class WordToVecDataset(Dataset):
    
    def __init__(self, 
        text: str | Text, 
        vocabulary: set[str],
        word_idx_dict: dict[str, int], 
        idx_word_dict: dict[int, str],
        window_size: int | None=3,
    ):
        self.text = text if isinstance(text, Text) else (
            Text(path=text) if Path(text).exists() else Text(text=text)
        )
        self.word_idx_dict = word_idx_dict
        self.idx_word_dict = idx_word_dict
        self.vocabulary = vocabulary
        self.window_size = window_size
    
        self.data = []
        self.get_pairs()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index][0], dtype=torch.long), torch.tensor(self.data[index][1], dtype=torch.long)
    
    
    def get_pairs(self):
        for i in range(self.text.text_length):
            for j in range(max(0, i - self.window_size), min(self.text.text_length, i + self.window_size + 1)):
                if (
                    i != j
                    and self.text.words[i] in self.word_idx_dict.keys()
                    and self.text.words[j] in self.word_idx_dict.keys()
                ):
                    self.data.append((
                        self.word_idx_dict.get(self.text.words[i]),
                        self.word_idx_dict.get(self.text.words[j])
                        )
                    )
                    
    def append(self, text: str):
        if Path(text).exists():
            self.text.text_from_file(path=text)
        else:
            self.text.text_from_string(text=text)
            
        self.get_pairs()