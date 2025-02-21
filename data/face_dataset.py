import torch
from torch.utils.data import Dataset
import numpy as np

class FaceSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # Shape: (N, seq_len, embedding_size)
        self.clarity_scores = sequences['clarity_scores']  # Shape: (N, seq_len)
        self.labels = labels  # Shape: (N,)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'embedding_sequence': torch.FloatTensor(self.sequences[idx]),
            'clarity_scores': torch.FloatTensor(self.clarity_scores[idx]),
            'label': torch.LongTensor([self.labels[idx]])
        }
