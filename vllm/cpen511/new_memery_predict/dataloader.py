
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from util import sequence_to_onehot


class SequenceDataset(Dataset):
    def __init__(self, filepath, windows_size):
        """
        Custom PyTorch Dataset to load sequences from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
        """
        self.data = pd.read_csv(filepath, usecols=["sequence_id"]).values  # Extract values as NumPy array
        self.data = torch.tensor(self.data, dtype=torch.float32).squeeze()  # 
        self.windows_size = windows_size

    def __len__(self):
        """Returns the number of valid sequences in the dataset."""
        return len(self.data) - self.windows_size

    # get the sequence of the access and label
    def __getitem__(self, idx):
        """Retrieves a sequence as a PyTorch tensor."""
        if(idx + self.windows_size + 1 > len(self.data)):
            return None, None
        sequence = self.data[idx:idx + self.windows_size]
        label = self.data[idx + self.windows_size]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
    
class SequenceDatasetOneHot(Dataset):
    def __init__(self, filepath, windows_size, mod):
        """
        Custom PyTorch Dataset to load sequences from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
        """
        self.data = pd.read_csv(filepath, usecols=["sequence_id"]).values  # Extract values as NumPy array
        self.data = torch.tensor(self.data, dtype=torch.float32).squeeze()  # 
        self.windows_size = windows_size
        self.mod = mod

    def __len__(self):
        """Returns the number of valid sequences in the dataset."""
        return len(self.data) - self.windows_size

    # get the sequence of the access and label
    def __getitem__(self, idx):
        """Retrieves a sequence as a PyTorch tensor."""
        if(idx + self.windows_size + 1 > len(self.data)):
            return None, None
        sequence = self.data[idx:idx + self.windows_size]
        label = self.data[idx + self.windows_size]
        sequence_onehot = sequence_to_onehot(self.mod, sequence)
        label_onehot = sequence_to_onehot(self.mod, label)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), sequence_onehot, label_onehot