
import os
import sys

import torch
import torch.optim as optim
from dataloader import SequenceDataset, SequenceDatasetOneHot
from model import (EarlyStopping, LstmMemoryPredict, MlpMemoryPredict,
                   MlpMemoryPredictClassification)
from util import onehot_to_sequence, sequence_to_onehot

window_size = 4
mod = 100
data_file_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/output.csv"
output_log = os.path.dirname(os.path.abspath(__file__)) + "/../data/output.txt"

# Redirect output
result_output = open(output_log, "w", buffering=1)
sys.stdout = result_output


# Split into train and test
data = SequenceDatasetOneHot(data_file_path , window_size, mod)
train_size = int(len(data) * 0.75)
train, test = torch.utils.data.random_split(data, [train_size, len(data) - train_size])

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=2, shuffle=False)


# Initialize model
model = MlpMemoryPredictClassification(window_size=window_size, num_classes=mod).cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Lost function
def loss_fn(input, outputs, labels):
    average = input.mean()
    diff_out = labels - outputs
    diff_label = labels - average
    return criterion(diff_out, diff_label)
    
# Train the model
for epoch in range(20):
    model.train()
    early_stopping = EarlyStopping(patience=3)
    for inputs, label, input_onehot, label_onehot in train_loader:
        input_onehot, label_onehot = input_onehot.cuda(), label_onehot.cuda()
        optimizer.zero_grad()
        outputs = model(input_onehot)
        loss = criterion(outputs, label_onehot)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")
    

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, label, input_onehot, label_onehot in test_loader:
            input_onehot, label_onehot = input_onehot.cuda(), label_onehot.cuda()
            outputs = model(input_onehot)
            loss = criterion(outputs, label_onehot)
            val_loss += loss.item()
    val_loss /= len(test_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
    
    schedular.step()
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
            print("Early stopping")
            break

    
# Testing
model.eval()
test_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for inputs, label, input_onehot, label_onehot in test_loader:
        inputs, label = inputs.cuda(), label.cuda()
        input_onehot, label_onehot = input_onehot.cuda(), label_onehot.cuda()
        outputs = model(input_onehot)
        loss = criterion(outputs, label_onehot)
        outputs = onehot_to_sequence(outputs)
        test_loss += loss.item()
        total += label.size(0)
        correct += (torch.round(outputs) == label%mod).sum().item()
        print(f"Output: {outputs.tolist()}, Label: {(label%mod).tolist()}, correct: {correct}, total: {total}")
print(f"Test Loss: {test_loss / len(test_loader)}, Accuracy: {(correct / total) * 100}%")

