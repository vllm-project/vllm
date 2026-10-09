# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compensated BF16 mHC prenorm with DeepGEMM on Hopper.

Run with the repository's .venv/bin/python. Reports prenorm only, excluding
weight preparation, allocations and the common Sinkhorn/RMSNorm epilogue.
No serving-level speedup can be inferred without an integrated benchmark.
"""

import argparse
import json
import statistics
from functools import partial

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.mhc.bf16x3 import (
    mhc_prenorm_bf16x3,
    split_bf16_mhc_weight,
)
from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[32, 64, 128, 512, 4096]
    )
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (9, 0), "Hopper benchmark"
    torch.manual_seed(0)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}))
    for m in args.tokens:
        assert m > 0
        k, n = 20480, 24
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda") / k**0.5
        parts = split_bf16_mhc_weight(weight)
        reference = x.double() @ weight.double().T
        square_reference = x.double().square().sum(-1)
        for backend in ("deepgemm", "bf16x3"):
            results = []
            for splits in (1, 4, 16):
                mix = torch.empty(splits, m, n, device="cuda")
                sq = torch.empty(splits, m, device="cuda")
                for block_m in (32, 64, 128) if backend == "bf16x3" else (0,):
                    run = (
                        partial(mhc_prenorm_bf16x3, x, parts, mix, sq, block_m=block_m)
                        if backend == "bf16x3"
                        else partial(tf32_hc_prenorm_gemm, x, weight, mix, sq, splits)
                    )

                    run()
                    torch.testing.assert_close(
                        mix.sum(0).double(), reference, rtol=5e-5, atol=2e-5
                    )
                    torch.testing.assert_close(
                        sq.sum(0).double(), square_reference, rtol=2e-6, atol=1e-5
                    )
                    for _ in range(10):
                        run()
                    torch.accelerator.synchronize()
                    samples = bench_gpu_time_with_cupti(
                        run, use_cuda_graph=True, cold_l2_cache=True
                    )
                    us = statistics.median(samples) * 1e3
                    row = dict(
                        backend=backend,
                        m=m,
                        k=k,
                        n=n,
                        splits=splits,
                        block_m=block_m,
                        median_us=us,
                        min_us=min(samples) * 1e3,
                        max_us=max(samples) * 1e3,
                        # Useful projection FLOPs, not the three component GEMMs.
                        useful_tflops=2 * m * n * k / (us * 1e6),
                    )
                    results.append(row)
                    print(json.dumps(row))
            print(json.dumps({"best": min(results, key=lambda row: row["median_us"])}))


if __name__ == "__main__":
    main()
