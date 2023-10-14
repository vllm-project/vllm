import torch
from vllm import topk as vllm_topk
base_list=[i for i in range(99)]
data=[base_list for _ in range(9)]

# data use for test
data=torch.Tensor(data).to(dtype=torch.float).cuda()

# tensor for storing the results
logits=torch.Tensor(data.shape[-1]).to(dtype=torch.float).cuda()

# different top_ks and top_ps
top_ks=[1,2,2,2,2,3,3,3,3,6 ]
top_ps=[-1.8]*5+[0.9]*5

vllm_topk.top_k(data,logits,top_ks,top_ps)
print(logits)
