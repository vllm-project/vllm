# input [batch_size, windows_size] or [windows_size]
# output [batch_size, windows_size, mod] or [windows_size, mod]
import torch
import torch.nn as nn


def sequence_to_onehot(mod, input):
    is_batched = input.dim() == 2  # Check if input has a batch dimension
    is_label = input.dim() == 0  # Check if input is a label
    if not is_batched:
        if(input.dim() == 1):
            input = input.unsqueeze(0)
        elif(input.dim() == 0):
            input = input.unsqueeze(0).unsqueeze(0)  # Temporarily add batch dimension

    batch_size, window_size = input.shape  # Get dimensions
    onehot = torch.zeros(batch_size, window_size, mod, dtype=torch.float32, device=input.device)
    onehot.scatter_(2, (input%mod).unsqueeze(2).long(), 1)  

    if not is_batched:
        if(not is_label):
            onehot = onehot.squeeze(0)
        elif(is_label):
            onehot = onehot.squeeze(0).squeeze(0)  # Remove batch dimension
    
    return onehot

# guess the most significant digits
# input: [windows_size, mod]
# output: [batch_size, windows_size]
# guess the most significant digits by voting
    
    


def onehot_to_sequence(input_onehot):
    is_batched = input_onehot.dim() == 3  # Check if input has a batch dimension
    if not is_batched:
        input_onehot = input_onehot.unsqueeze(0)  # Temporarily add batch dimension

    batch_size, window_size, _ = input_onehot.shape  # Get dimensions
    sequence = torch.argmax(input_onehot, dim=2)  # Convert one-hot to sequence
    if not is_batched:
        sequence = sequence.squeeze(0)  # Remove batch dimension
    return sequence
    