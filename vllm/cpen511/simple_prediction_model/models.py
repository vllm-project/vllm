import torch.nn as nn
import torch

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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # lstm
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        # initial hidden state
        self.lstm_hidden = torch.zeros(1, 1, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out, self.lstm_hidden = self.lstm(out, self.lstm_hidden)
        # out = self.dropout(out)
        out = self.fc2(out)
        return out