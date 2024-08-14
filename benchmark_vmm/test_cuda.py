import torch
print(torch.randn(1).cuda())
print(torch.rand(5, 3, device=torch.device('cuda')))

