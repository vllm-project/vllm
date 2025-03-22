import torch
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as Fun

class sequenceDataset(Dataset):
    def __init__(self, sequence, lookback, lookforward):
        self.sequence = torch.tensor(sequence, dtype=torch.float32)
        self.lookback = lookback
        self.lookforward = lookforward
        
    def __len__(self):
        return len(self.sequence) - self.lookback - self.lookforward

    def __getitem__(self, idx):
        history = self.sequence[idx:idx+self.lookback].clone().detach()
        future = self.sequence[idx+self.lookback:idx+self.lookback+self.lookforward].clone().detach()
        return (history, future)
        
def load_data(data, batch_size, lookback, lookforward):
    data = data.copy()
    sequence = data['sequence_id'].values
    dataset = sequenceDataset(sequence, lookback, lookforward)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
