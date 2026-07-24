"""
This script benchmarks the cutedsl kernel against the cutlass kernel for 
various input sizes and configurations. It uses triton.do_bench tool for benchmarking.
It generates random input tensors, runs both kernels multiple times, and measures 
their execution time. The results are printed in a tabular format, showing the 
average execution time for each kernel and the speedup achieved by the cutedsl 
kernel over the cutlass kernel.
"""
import copy
import math
from dataclasses import dataclass
from itertools import product
from typing import Any

import torch
import triton

from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.quantization.cutedsl.scaled_mm_dispatch import (
    cutedsl_scaled_mm,
)
from vllm.platforms import current_platform


@dataclass
class Row:
    case: str
    cutedsl_ms: float
    cutlass_ms: float
    speedup_x: float


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    m_size_list = [1, 2, 4, 8, 16, 32, 64, 128]
    b_shape_list = [
        # qwen3-1.7B
        # TP=1
        # (2048, 4096),
        # (2048, 2048),
        # (2048, 12288),
        # (6144, 2048),
        # # qwen3-8B
        # # TP=1
        # (4096, 6144),
        (4096, 4096),
        # (4096, 24576),
        # (12288, 4096),
        # # qwen3-32B
        # # TP=1
        # (5120, 10240),
        # (5120, 5120),
        # (5120, 51200),
        # (25600, 5120),
    ]

    in_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.bfloat16
    inputs = {}
    for M, (K, N) in product(m_size_list, b_shape_list):
        scale = 1.0 / math.sqrt(K)
        a = (scale * (0.5 + torch.rand(M, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = (scale * (0.5 + torch.rand(N, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = b.t()
        c = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # per token
        scale_a = 0.5 + torch.rand((1, M), dtype=scale_dtype, device="cuda")
        scale_a = scale_a.t()
        # per tensor
        # scale_a = 0.5 + torch.rand((1, 1), dtype=scale_dtype, device="cuda")
        # per tensor
        scale_b = 0.5 + torch.rand((1, 1), dtype=scale_dtype, device="cuda")
        bias = 0.5 * (torch.rand(N, dtype=out_dtype, device="cuda") - 0.5)

        config_key = CaseKey(
            {
                "K": K,
                "N": N,
                "M": M,
            }
        )
        inputs[config_key] = (c, a, b, scale_a, scale_b, bias)

    return inputs


def baseline_cutedsl(
    c: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
):
    return cutedsl_scaled_mm(a, b, scale_a, scale_b, c.dtype, bias)


def baseline_cutlass(
    c: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
):
    # re-initialize tensor c to get a fair comparison with cutlass
    c = torch.empty((a.shape[0], b.shape[1]), dtype=c.dtype, device=a.device)
    torch.ops._C.cutlass_scaled_mm(c, a, b, scale_a, scale_b, bias)
    return c


def print_table(rows: list[Row]) -> None:
    headers = [
        "case",
        "cutedsl_ms",
        "cutlass_ms",
        "speedup(x)",
    ]

    data = [
        [
            r.case,
            f"{r.cutedsl_ms:.3f}",
            f"{r.cutlass_ms:.3f}",
            f"{r.speedup_x:.3f}",
        ]
        for r in rows
    ]

    cols = list(zip(*([headers] + data)))
    widths = [max(len(cell) for cell in col) for col in cols]

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))


def cleanup_gpu_resources():
    import gc

    try:
        if torch.cuda.is_available():
            # Clear GPU memory cache
            torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Clear torch compilation cache
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()

            # Synchronize all CUDA streams
            torch.cuda.synchronize()

            # Reset peak memory stats for clean measurements
            torch.cuda.reset_peak_memory_stats()

            print("GPU resources cleaned up successfully")

    except Exception as e:
        print(f"Failed to cleanup GPU resources: {e}")


@torch.inference_mode()
def benchmark(cutlass_kernel, cutedsl_kernel, repeat=1000, cudagraph=True):
    rows: list[Row] = []
    benchmark_fn = (
        triton.testing.do_bench_cudagraph if cudagraph else triton.testing.do_bench
    )

    inputs_dict = generate_inputs()

    for key, inputs in inputs_dict.items():
        try:
            print(f"Start benchmarking with key {key}")

            inputs_clone = copy.deepcopy(inputs)

            cutlass_fn = lambda: cutlass_kernel(*inputs)
            cutedsl_fn = lambda: cutedsl_kernel(*inputs_clone)

            cutlass_latency = benchmark_fn(cutlass_fn, rep=repeat, return_mode="mean")

            cutedsl_latency = benchmark_fn(
                cutedsl_fn, rep=repeat, return_mode="mean"
            )

            speedup = cutedsl_latency / cutlass_latency

            rows.append(
                Row(
                    case=str(key),
                    cutedsl_ms=cutedsl_latency,
                    cutlass_ms=cutlass_latency,
                    speedup_x=speedup,
                )
            )

            cleanup_gpu_resources()

        except Exception as e:
            raise e
            print(f"Benchmarking failed for key {key}: {e}")
            continue

    print_table(rows)


if __name__ == "__main__":
    benchmark(baseline_cutlass, baseline_cutedsl)
