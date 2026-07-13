# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: fused softcap Triton kernel vs PyTorch 3-op baseline."""

import time

import torch

from vllm.model_executor.layers.softcap_kernel import softcap_logits


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


cap = 30.0
print(
    f"{'Shape':>20s} | {'PyTorch (ms)':>12s} | {'Triton (ms)':>12s} | {'Speedup':>8s}"
)
print("-" * 65)

for batch in [1, 8, 32, 64, 128]:
    for vocab in [32000, 128000, 256000]:
        shape = (batch, vocab)
        logits = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        t_pt = bench(lambda logits=logits: cap * torch.tanh(logits / cap))
        t_tr = bench(lambda logits=logits: softcap_logits(logits.clone(), cap))
        print(
            f"{str(shape):>20s} | {t_pt:>12.3f} | {t_tr:>12.3f} | {t_pt / t_tr:>7.1f}x"
        )
