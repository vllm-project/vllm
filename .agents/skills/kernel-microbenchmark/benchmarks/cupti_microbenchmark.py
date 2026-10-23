# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlashInfer CUPTI microbenchmark template with throughput metrics."""

import statistics

import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

WARMUP = 25
MATMUL_CASES = [
    ("compute-bound", 4096, 4096, 4096),
    ("small-M", 16, 16384, 8192),
]


def bench_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.accelerator.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn)) * 1e3


def main() -> None:
    if not torch.accelerator.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA is required for CUPTI kernel timing.")

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    rows = []
    for name, m, n, k in MATMUL_CASES:
        a = torch.randn(m, k, dtype=torch.bfloat16)
        b = torch.randn(k, n, dtype=torch.bfloat16)
        out = torch.empty(m, n, dtype=torch.bfloat16)

        def run_matmul(a=a, b=b, out=out):
            torch.mm(a, b, out=out)

        run_matmul()
        ref = torch.mm(a, b)
        torch.accelerator.synchronize()
        torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)

        us = bench_us(run_matmul)
        rows.append(
            {
                "case": name,
                "shape": f"{m}x{n}x{k}",
                "us": us,
                "tflops": 2 * m * n * k / (us * 1e6),
                "gbps": 2 * (m * k + k * n + m * n) / (us * 1e3),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
