
import os
import sys

import torch
import torch.optim as optim
from dataloader import SequenceDataset
from model import EarlyStopping, LstmMemoryPredict, MlpMemoryPredict

windows_size = 8
data_file_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/output.csv"
output_log = os.path.dirname(os.path.abspath(__file__)) + "/../data/output.txt"

# Redirect output
result_output = open(output_log, "w", buffering=1)
sys.stdout = result_output


# Split into train and test
data = SequenceDataset(data_file_path , windows_size)
train_size = int(len(data) * 0.75)
train, test = torch.utils.data.random_split(data, [train_size, len(data) - train_size])

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=2, shuffle=False)


# Initialize model
model = LstmMemoryPredict(windows_size=windows_size).cuda()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Lost function
def loss_fn(input, outputs, labels):
    average = input.mean()
    diff_out = labels - outputs
    diff_label = labels - average
    return criterion(diff_out, diff_label)
    
# Train the model
for epoch in range(50):
    model.train()
    early_stopping = EarlyStopping(patience=3)
    for inputs, label in train_loader:
        inputs, label = inputs.cuda(), label.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")
    

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs, label = inputs.cuda(), label.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, label)
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
    for inputs, label in test_loader:
        inputs, label = inputs.cuda(), label.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        test_loss += loss.item()
        total += label.size(0)
        correct += (torch.round(outputs) == label).sum().item()
        print(f"Output: {outputs.tolist()}, Label: {label.tolist()}, correct: {correct}, total: {total}")
print(f"Test Loss: {test_loss / len(test_loader)}, Accuracy: {(correct / total) * 100}%")
result_output

