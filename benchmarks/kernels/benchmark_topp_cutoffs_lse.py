# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark topp_cutoffs_lse against the top-p path it replaces in verification.

The baseline is what MRV2's rejection sampler runs today for top-p requests:
an fp32 copy of the bf16 target logits, the temperature kernel and the Triton
top-k/top-p mask. topp_cutoffs_lse reads the bf16 logits once and writes one
cutoff and one log-sum-exp per row.

    .venv/bin/python benchmarks/kernels/benchmark_topp_cutoffs_lse.py
"""

import argparse
import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature


def bench_us(fn) -> float:
    for _ in range(10):
        fn()
    torch.accelerator.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn)) * 1e3


def main(args) -> None:
    torch.manual_seed(0)
    print(f"{'vocab':>7} {'rows':>5} {'baseline us':>12} {'op us':>8} {'op GB/s':>8}")
    for vocab in args.vocab:
        for rows in args.rows:
            # A Gaussian body under a few dominant tokens, like LM logits. Flat
            # rows (large-sigma Gaussians) sometimes take the fp64 path instead.
            logits = torch.randn(rows, vocab, device="cuda") * 2
            spikes = torch.randint(0, vocab, (rows, 4), device="cuda")
            logits.scatter_(1, spikes, 12 + torch.rand(rows, 4, device="cuda") * 6)
            logits = logits.to(torch.bfloat16)
            idx = torch.arange(rows, dtype=torch.int32, device="cuda")
            temp = torch.full((rows,), args.temperature, device="cuda")
            top_p = torch.full((rows,), args.top_p, device="cuda")

            def baseline(logits=logits, idx=idx, temp=temp, top_p=top_p):
                x = logits.float()
                apply_temperature(x, idx, temp)
                apply_top_k_top_p(x, None, top_p)

            def op(logits=logits, idx=idx, temp=temp, top_p=top_p):
                ops.topp_cutoffs_lse(logits, idx, temp, top_p)

            base_us, op_us = bench_us(baseline), bench_us(op)
            gbps = rows * vocab * logits.element_size() / (op_us * 1e-6) / 1e9
            print(f"{vocab:>7} {rows:>5} {base_us:>12.1f} {op_us:>8.1f} {gbps:>8.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, nargs="+", default=[151936, 202048])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    main(parser.parse_args())
