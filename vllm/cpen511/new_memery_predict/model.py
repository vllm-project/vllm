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
    

class TransformerMemoryPredict(nn.Module):
    def __init__(self, windows_size):
        super(TransformerMemoryPredict, self).__init__()
        self.transformer = nn.Transformer(d_model=windows_size, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512)
        self.fc = nn.Linear(windows_size, 1, bias=True)
        
    # takes a sequence of memory access and predicts the next memory access
    # x: [batch_size, windows_size]
    # output: [batch_size]
    def forward(self, x):
        out = self.transformer(x, x)
        out = self.fc(out)
        return out.squeeze(1)
    
import torch
import torch.nn as nn


class MlpMemoryPredictClassification(nn.Module):
    def __init__(self, window_size, num_classes):
        super(MlpMemoryPredictClassification, self).__init__()
        self.fc1 = nn.Linear(window_size * num_classes, 32, bias=True)  # Fixed input size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, num_classes, bias=True)  # Fixed layer name

    # Takes a sequence of memory accesses and predicts the next memory access class
    # x: [batch_size, window_size, num_classes]
    # output: [batch_size, num_classes] (logits for classification)
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten [batch_size, window_size, num_classes] -> [batch_size, window_size * num_classes]
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)  # Fixed incorrect reference
        out = torch.nn.functional.log_softmax(out, dim=1)

        return out  # No need for squeeze(1) since this is a classification model
