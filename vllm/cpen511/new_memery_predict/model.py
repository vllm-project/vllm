# goal predict the next memory access (delta_in) given history of memory accesses (pc)

import torch
import torch.nn as nn


class MlpMemoryPredict(nn.Module):
    def __init__(self, windows_size):
        super(MlpMemoryPredict, self).__init__()
        self.fc1 = nn.Linear(windows_size, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1, bias=True)
        
    # takes a sequence of memory access and predicts the next memory access
    # x: [batch_size, windows_size]
    # output: [batch_size]
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(1)
    
           
    
class LstmMemoryPredict(nn.Module):
    def __init__(self, windows_size):
        super(LstmMemoryPredict, self).__init__()
        self.lstm = nn.LSTM(windows_size, 128, 1, batch_first=True)
        self.fc = nn.Linear(128, 1, bias=True)
        
    # takes a sequence of memory access and predicts the next memory access
    # x: [batch_size, windows_size]
    # output: [batch_size]
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(1)