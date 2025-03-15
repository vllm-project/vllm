# goal predict the next memory access (delta_in) given history of memory accesses (pc)

import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, epoch=114514):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class MlpMemoryPredict(nn.Module):
    def __init__(self, windows_size):
        super(MlpMemoryPredict, self).__init__()
        self.fc1 = nn.Linear(windows_size, 32, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.f3 = nn.Linear(32, 1, bias=True)
        
    # takes a sequence of memory access and predicts the next memory access
    # x: [batch_size, windows_size]
    # output: [batch_size]
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.f3(out)
        return out.squeeze(1)
    
           
    
class LstmMemoryPredict(nn.Module):
    def __init__(self, windows_size):
        super(LstmMemoryPredict, self).__init__()
        self.lstm = nn.LSTM(input_size=windows_size, hidden_size=32, num_layers=5, batch_first=True, bias=True)
        self.fc = nn.Linear(32, 1, bias=True)
        
    # takes a sequence of memory access and predicts the next memory access
    # x: [batch_size, windows_size]
    # output: [batch_size]
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(1)