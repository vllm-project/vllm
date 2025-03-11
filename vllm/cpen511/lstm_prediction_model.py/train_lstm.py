import sys, os


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

from config import *
# keep the path
path_keeper['targetpath'] = targetpath
path_keeper['filepath'] = filepath

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from load_trace import load_traces
# Load the traces
train_data, test_data = load_traces(filepath, shuffle=False)

from label_encoder import encode_data
# Encode the data
train_iter, test_iter, target_keys = encode_data(train_data, test_data)

from lstm import EmbeddingLSTM
# Create the model
my_model = EmbeddingLSTM(
    config['num_pc'],
    config['num_delta_in'],
    config['num_output_next'],
    hparams["embed_dim"],
    hparams["hidden_dim"],
    hparams["output_dim"],
    topPredNum=hparams["topPredNum"],
    num_layers=hparams["num_layers"],
    dropout=hparams["dropout"]
).to(device)


# Prepare for training
from torch.optim.lr_scheduler import ExponentialLR
train_loss = []
optimizer = torch.optim.Adam(my_model.parameters(), lr=hparams["learning_rate"])
scheduler = ExponentialLR(optimizer, gamma=0.8)
batch_size = config['batch_size']
with open(targetpath + "/loss.txt", "w") as f:
    f.write(f"Training 75% traces\n")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, epoch=114514):
        print(f"Epoch {epoch}, Epoch loss: {val_loss:.8f}")
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
# Initialize early stopping
early_stopping = EarlyStopping(patience=5, min_delta=0.01)

from validation import validate_model
for epoch in range(hparams["epochs"]):
    my_model.train()
    h_0 = torch.zeros(hparams["num_layers"], batch_size, hparams["hidden_dim"]).to(device)
    c_0 = torch.zeros(hparams["num_layers"], batch_size, hparams["hidden_dim"]).to(device)
    lstm_state = (h_0, c_0)

    for idx, batch in enumerate(train_iter):
        # torch.cuda.empty_cache()
        batch = [ds.to(device) for ds in batch]
        inputs = batch[:-1]
        targets = batch[-1]
        _, lstm_state, batch_loss = my_model(inputs, lstm_state, targets)
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=1)
        optimizer.step()

        # train_loss.append(float(batch_loss))
        if idx % 4096 == 0:
        # if idx % 32 == 0:
            with open(targetpath + "/loss.txt", "a") as f:
                if idx % 16384 == 0 and idx != 0:
                # if idx % 128 == 0:
                # if idx % 512 == 0:
                    parts = 0.0001
                    current_accuracy_10 = validate_model(my_model, test_iter, target_keys, computing_device=device, initial_state=lstm_state, parts=parts)
                    f.write(f"Epoch {epoch + 1}, Iteration {hex(idx)}, Loss: {batch_loss:.8f}, Accuracy_10: {current_accuracy_10:.4f} in first {parts*100}% of the testing data\n")
                    my_model.train()
                else:
                    f.write(f"Epoch {epoch + 1}, Iteration {hex(idx)}, Loss: {batch_loss:.8f}\n")
                
            
        lstm_state = tuple([s.data for s in lstm_state])
    scheduler.step()
    # for each epoch, save a tmp model
    torch.save(my_model.state_dict(), targetpath + f"/LSTM_model_tmp_{epoch}.pt")
    
    # Check early stopping criteria
    early_stopping(batch_loss, epoch)
    if early_stopping.early_stop:
        print("Early stopping")
        
    
# save the model for this fold
torch.save(my_model.state_dict(), targetpath + f"/LSTM_model.pt")


# Evaluate the model on the test set for the current fold
fold_accuracy_10 = validate_model(my_model, test_iter, target_keys, computing_device=device)
print(f"accuracy_10: {fold_accuracy_10}")

result_output.close()