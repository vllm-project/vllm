import sys, os
import pandas as pd

# Get the arguments from the command line
arguments = sys.argv[1:]
filepath = arguments[0]
targetpath = arguments[1]

Good = True
if filepath == None:
    print("Please provide the path to the trace file")
    Good = False

if targetpath == None:
    print("Please provide the path to the target output file")
    Good = False
    
if not Good:
    sys.exit(1)

# make them absolute paths
if not os.path.isabs(filepath):
    filepath = os.path.abspath(filepath)
print("Absolute path to trace file: ", filepath)
    
if not os.path.isabs(targetpath):
    targetpath = os.path.abspath(targetpath)
print("Absolute path to target output file: ", targetpath)

print("Redirecting output to ", targetpath + "/output.txt")
result_output = open(targetpath + "/output.txt", "w", buffering=1)
sys.stdout = result_output
print("Redirected output to ", targetpath + "/output.txt")

# load the traces
trace_file = pd.read_csv(filepath)
# trace_file = trace_file['sequence_id'].values

# configs
look_back = 8
look_forward = 4
batch_size = 1
hidden_dim = 32
epochs = 20

# split into 75% train and 25% test
train_size = int(len(trace_file) * 0.75)
test_size = len(trace_file) - train_size
train, test = trace_file[0:train_size], trace_file[train_size:len(trace_file)]
print(f"train size: {len(train)}, test size: {len(test)}")

from data_loader import *
# prepare the data
train_loader = load_data(train, batch_size, look_back, look_forward)
test_loader = load_data(test, batch_size, look_back, look_forward)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from models import *
from sklearn.model_selection import KFold
# create the model
model = MLP(look_back, hidden_dim, look_forward).to(device)

# Prepare the optimizer
import torch.optim as optim
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 5-fold cross-validation
kf = KFold(n_splits=5)
fold = 1

for train_index, val_index in kf.split(train):
    print(f"Fold {fold}")
    fold += 1

    train_fold, val_fold = train.iloc[train_index], train.iloc[val_index]
    train_loader = load_data(train_fold, batch_size, look_back, look_forward)
    val_loader = load_data(val_fold, batch_size, look_back, look_forward)

    # need to clear early stopping for each fold
    early_stopping = EarlyStopping(patience=3)
    for epoch in range(epochs):
        model.train()
        for batch in (train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(inputs, outputs, targets)
            loss = criterion(outputs, targets)
            print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        schedular.step()
        # early_stopping(val_loss, model)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        
# Test the model
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    prediction_output = open(targetpath + "/predictions.txt", "w", buffering=1)
    for batch in test_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        # Print predictions and labels
        for i in range(len(outputs)):
            prediction_output.write(f"Prediction: {outputs[i].cpu().numpy()}, Label: {targets[i].cpu().numpy()}\n")
            # change all to closest integer
            outputs[i] = torch.round(outputs[i])
            targets[i] = torch.round(targets[i])
            for label in targets[i]:
                if label in outputs[i]:
                    correct += 1
                total += 1
    prediction_output.close()
        
# Save the model
torch.save(model.state_dict(), targetpath + "/model.pth")
print(f"Test Loss: {test_loss / len(test_loader)}")
print(f"Accuracy: {correct / total}")
result_output.close()
sys.exit(0)
