import torch
from torch.profiler import profile, record_function, ProfilerActivity

x = torch.randn(512, 512).cuda()
y = torch.randn(512, 512).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    x[...] = x

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    x.copy_(x)

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    y[...] = x

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    y.copy_(x)

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))