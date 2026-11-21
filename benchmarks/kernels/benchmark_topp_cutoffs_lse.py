# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark topp_cutoffs_lse against the top-p path it replaces in verification.

The baseline is what MRV2's rejection sampler runs today for top-p requests:
an fp32 copy of the bf16 target logits, the temperature kernel and the Triton
top-k/top-p mask. topp_cutoffs_lse reads the bf16 logits once and writes one
cutoff and one log-sum-exp per row; "op" lets it pick the cluster size, "C=4"
forces four CTAs per row. --force-fallback times the precise (1) or fp64 (2)
path on every row.

    .venv/bin/python benchmarks/kernels/benchmark_topp_cutoffs_lse.py
"""

import argparse
import statistics
from functools import partial

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
    print(
        f"{'vocab':>7} {'rows':>5} {'baseline us':>12} {'C=4 us':>8} {'op us':>8} "
        f"{'op GB/s':>8}"
    )
    for vocab in args.vocab:
        for rows in args.rows:
            # A Gaussian body under a few dominant tokens, like LM logits. Flat
            # rows (large-sigma Gaussians) sometimes take the fp64 path instead.
            logits = torch.randn(rows, vocab, device="cuda") * 2
            spikes = torch.randint(0, vocab, (rows, 4), device="cuda")
            logits.scatter_(1, spikes, 12 + torch.rand(rows, 4, device="cuda") * 6)
            logits = logits.to(torch.bfloat16)
            idx = torch.arange(rows, dtype=torch.int64, device="cuda")
            temp = torch.full((rows,), args.temperature, device="cuda")
            top_p = torch.full((rows,), args.top_p, device="cuda")

            def baseline(logits=logits, idx=idx, temp=temp, top_p=top_p):
                x = logits.float()
                apply_temperature(x, idx, temp)
                apply_top_k_top_p(x, None, top_p)

            def op(logits=logits, idx=idx, temp=temp, top_p=top_p, c=0):
                ops.topp_cutoffs_lse(
                    logits, idx, temp, top_p, c, force_fallback=args.force_fallback
                )

            base_us, op_us = bench_us(baseline), bench_us(op)
            c4_us = bench_us(partial(op, c=4)) if vocab <= 4 * 65528 else float("nan")
            gbps = rows * vocab * logits.element_size() / (op_us * 1e-6) / 1e9
            print(
                f"{vocab:>7} {rows:>5} {base_us:>12.1f} {c4_us:>8.1f} {op_us:>8.1f} "
                f"{gbps:>8.0f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, nargs="+", default=[151936, 202048])
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 16, 32, 64, 128, 256, 512, 1024]
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--force-fallback", type=int, default=0, choices=[0, 1, 2])
    main(parser.parse_args())
