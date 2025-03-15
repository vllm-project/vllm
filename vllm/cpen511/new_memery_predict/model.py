# goal predict the next memory access (delta_in) given history of memory accesses (pc)

import torch
import torch.nn as nn


class MemoryPredict(nn.Module):
    def __init__(self, windows_size):
        super(MemoryPredict, self).__init__()
        self.fc1 = nn.Linear(windows_size, 128, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 1, bias=True)
        
    # takes a sequence of memory access and predicts the next memory access
    # x: [batch_size, windows_size]
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    

    def backward(self, x):
        # Implement the backward pass if needed
        pass
        
    def loss_function(self, output, target):
        return torch.nn.MSELoss()(output, target)
        
    
    