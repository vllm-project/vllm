# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time token->request mapping. Run this on main and on the PR."""

from functools import partial

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices

a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(100):
    a = a @ a
torch.accelerator.synchronize()

print(torch.cuda.get_device_name(0), torch.__version__, triton.__version__)
print("reqs,tokens_per_req,us")
for num_reqs, tpr in [(128, 5), (512, 5), (1024, 5), (1024, 64)]:
    n = num_reqs * tpr
    qsl = (
        torch.arange(num_reqs + 1, device="cuda", dtype=torch.int32) * tpr
    ).contiguous()
    out = torch.empty(n, device="cuda", dtype=torch.int32)
    fn = partial(compute_token_to_req_indices, qsl, out, n, n)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            fn()
    torch.accelerator.synchronize()
    us = (
        triton.testing.do_bench(graph.replay, warmup=100, rep=300, return_mode="median")
        / 100
    )
    print(f"{num_reqs},{tpr},{us * 1e3:.2f}", flush=True)
