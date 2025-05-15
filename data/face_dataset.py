import torch
from torch.utils.data import Dataset
import numpy as np

class FaceSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # List of dicts with 'embeddings' and 'clarity_scores'
        self.labels = labels  # List of ints
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'embedding_sequence': torch.FloatTensor(seq['embeddings']),
            'clarity_scores': torch.FloatTensor(seq['clarity_scores']),
            'label': torch.LongTensor([self.labels[idx]])
        }
