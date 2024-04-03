import torch

x = torch.rand((2000)).cuda()
xx = torch.topk(x, k=300).indices
x = torch.rand((1000)).cuda()
xx = torch.topk(x, k=400).indices

for i in range(1000):
    x = torch.rand((4000)).cuda()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()    
    xx = torch.topk(x, k=500).indices
    end.record()
    torch.cuda.synchronize()
    temp_time = start.elapsed_time(end)
    print(temp_time)


print("finished")