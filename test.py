import torch
import pdb
out = torch.load("output.pt")
key = torch.load("key.pt")
value = torch.load("value.pt")
h = torch.load("h.pt")

check_out = torch.load("check_output.pt")
check_key = torch.load("check_key.pt")
check_value = torch.load("check_value.pt")
check_h = torch.load("check_h.pt")

pdb.set_trace()

print("finished")