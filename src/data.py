"""
data.py
-------
Toy dataset (copy task) and utilities.
"""
import torch
from torch.utils.data import Dataset, DataLoader

class CopyTaskDataset(Dataset):
    """
    Generates sequences of integers and the target is an exact copy.
    Useful to test that the Transformer can learn identity mapping.
    """
    def __init__(self, vocab_size=50, seq_len=10, size=2000):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size
        self.data = torch.randint(2, vocab_size, (size, seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x.clone()
        return x, y

def get_loader(batch_size=32, **kwargs):
    ds = CopyTaskDataset(**kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
