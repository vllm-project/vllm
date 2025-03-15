import argparse
import os
import sys

import pandas as pd
import torch
import torch.optim as optim
from data_loader import load_data
from models import MLP
from sklearn.model_selection import KFold


# Define argument parser
# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple prediction model")
    parser.add_argument("--filepath", help="Path to the trace file", default="/root/vllm/vllm/cpen511/data/trace.csv")
    parser.add_argument("--targetpath", help="Path to the target output file", default="/root/vllm/vllm/cpen511/data/")
    return parser.parse_args()  # Ensure we return the parsed arguments

# Get command-line arguments
args = parse_args()
filepath = os.path.abspath(args.filepath)
targetpath = os.path.abspath(args.targetpath)

print("Absolute path to trace file: ", filepath)
print("Absolute path to target output file: ", targetpath)


# Redirect output
output_file = os.path.join(targetpath, "output.txt")
print("Redirecting output to", output_file)
result_output = open(output_file, "w", buffering=1)
sys.stdout = result_output

# Load the traces
trace_file = pd.read_csv(filepath)

# Configurations
look_back, look_forward, batch_size, hidden_dim, epochs = 8, 4, 1, 32, 20

# Split into train and test
train_size = int(len(trace_file) * 0.75)
train, test = trace_file[:train_size], trace_file[train_size:]
print(f"Train size: {len(train)}, Test size: {len(test)}")

# Prepare data loaders
train_loader = load_data(train, batch_size, look_back, look_forward)
test_loader = load_data(test, batch_size, look_back, look_forward)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize model
model = MLP(look_back, hidden_dim, look_forward).to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# K-Fold Cross Validation
kf = KFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(kf.split(train), 1):
    print(f"Fold {fold}")
    train_fold, val_fold = train.iloc[train_idx], train.iloc[val_idx]
    train_loader = load_data(train_fold, batch_size, look_back, look_forward)
    val_loader = load_data(val_fold, batch_size, look_back, look_forward)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = sum(criterion(model(inputs.to(device)), targets.to(device)).item() for inputs, targets in val_loader) / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
        scheduler.step()

# Testing
model.eval()
test_loss, correct, total = 0, 0, 0
prediction_file = os.path.join(targetpath, "predictions.txt")
with open(prediction_file, "w", buffering=1) as prediction_output:
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            for i in range(len(outputs)):
                prediction_output.write(f"Prediction: {outputs[i].cpu().numpy()}, Label: {targets[i].cpu().numpy()}\n")
                outputs[i], targets[i] = torch.round(outputs[i]), torch.round(targets[i])
                correct += sum(label in outputs[i] for label in targets[i])
                total += len(targets[i])

# Save model
torch.save(model.state_dict(), os.path.join(targetpath, "model.pth"))
print(f"Test Loss: {test_loss / len(test_loader)}")
print(f"Accuracy: {correct / total}")
result_output.close()
sys.exit(0)
