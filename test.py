# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

e = 8
problem_sizes = torch.tensor([
    [50,  8],   
    [0,   4],   
    [200, 16],  
    [75,  8],   
    [0,   4],   
    [120, 12],  
    [0,   4],   
    [60,  8], 
], dtype=torch.int32)

m=50, num_topk=8, n=128, k=128
tokens_per_expert = problem_sizes[:,0]
round_up = lambda x, y: (x + y - 1) // y * y
sf_sizes = round_up(tokens_per_expert, 128)
sf_k = round_up(k // 16, 4)

inpput = torch.randn((m, k), dtype=torch.bfloat16, device='cuda:0')

#xhalf_k_w1 = torch.randn((n, k), dtype=dtype, device=device)


rep_a_fp4 = torch.empty(m*num_topk, k//2, dtype=torch.uint8, device=device)

rep_a_blockscale = torch.empty(sum(sf_sizes), sf_k, dtype=torch.float8_e4m3fn, device=device)
rep_a_gs = torch.empty(e,  dtype=torch.float32, device=device)
sf_offsets = torch.zeros(e+1, dtype=problem_sizes1.dtype, device=device)
sf_offsets[1:] = torch.cumsum(sf_sizes, dim=0) 
for expert_id in range(e):
    if tokens_per_expert[expert_id] == 0:
        continue
    sf_slice = slice(sf_offsets[expert_id],sf_offsets[expert_id+1])
    a_slice = slice(expert_offsets[expert_id], expert_offsets[expert_id+1]) 
    a_expert = rep_a[a_slice]
    a_expert_max = torch.abs(a_expert).max().to(torch.float32)
    rep_a_gs[expert_id] = 448.0 * 6.0 / a_expert_max
    rep_a_fp4[a_slice], rep_a_blockscale[sf_slice] = ops.scaled_fp4_quant(
                                            a_expert, rep_a_gs[expert_id])


