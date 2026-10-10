# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MRV2 min-p filtering; run on each checkout for a comparison.

Example:
    python benchmarks/kernels/benchmark_min_p.py --batch-sizes 1 8 32 128 512

"""

import argparse
from functools import partial

import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.sample.min_p import apply_min_p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512]
    )
    parser.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[32000, 128256, 151936]
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    print("batch,vocab,latency_us")
    for batch in args.batch_sizes:
        for vocab in args.vocab_sizes:
            logits = torch.randn(batch, vocab, device="cuda", dtype=torch.float32)
            mapping = torch.arange(batch, device="cuda", dtype=torch.int32)
            min_p = torch.full((batch,), 0.1, device="cuda")
            # Reapplying the filter is idempotent and preserves the row maximum.
            latency = triton.testing.do_bench_cudagraph(
                partial(apply_min_p, logits, mapping, min_p)
            )
            print(f"{batch},{vocab},{latency * 1000:.3f}")


if __name__ == "__main__":
    main()
