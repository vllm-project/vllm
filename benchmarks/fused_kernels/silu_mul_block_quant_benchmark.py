# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle as pkl
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from tqdm import tqdm

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.triton_quantization import (
    silu_and_mul_per_block_quant_triton,
)

@dataclass
class bench_params_t:
    num_tokens: int
    hidden_size: int
    dtype: torch.dtype
    group_size: int  # Changed from list[int] to int

    def description(self):
        return (
            f"N {self.num_tokens} "
            f"x D {self.hidden_size} "
            f"x DT {self.dtype} "
            f"x GS {self.group_size}"
        )


def get_bench_params() -> list[bench_params_t]:
    """Test configurations covering common model sizes."""
    NUM_TOKENS = [16, 128, 512, 2048]
    HIDDEN_SIZES = [1024, 2048, 4096, 5120, 14336]  # Common FFN sizes
    DTYPES = [torch.float16, torch.bfloat16]
    GROUP_SIZES = [64, 128]  # Changed from [[1, 64], [1, 128]]

    combinations = product(NUM_TOKENS, HIDDEN_SIZES, DTYPES, GROUP_SIZES)
    bench_params = list(
        map(lambda x: bench_params_t(x[0], x[1], x[2], x[3]), combinations)
    )
    return bench_params


# Reference implementations
def unfused_fp8_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,  # Changed from list[int]
):
    """Unfused: SiLU+Mul then per-tensor quantize."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    
    # SiLU(gate) * up
    silu_out = F.silu(gate) * up
    
    # Per-tensor quantize (no group_size used here)
    silu_out, _ = ops.scaled_fp8_quant(silu_out)


def unfused_groupwise_fp8_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,  # Changed from list[int]
):
    """Unfused: SiLU+Mul then group-wise quantize."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    
    # SiLU(gate) * up
    silu_out = F.silu(gate) * up
    
    # Group quantize - use group_size directly
    silu_out, _ = per_token_group_quant_fp8(
        silu_out, group_size=group_size, use_ue8m0=False
    )


def fused_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,
):
    """Fused: SiLU+Mul+Block Quantization in single kernel."""
    out, _ = ops.silu_and_mul_per_block_quant(
        x,
        group_size=group_size,
        quant_dtype=quant_dtype,
        is_scale_transposed=False,
    )


def triton_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,
):
    """Triton: SiLU+Mul+Block Quantization (Triton kernel)."""
    silu_and_mul_per_block_quant_triton(
        x,
        group_size=group_size,
        quant_dtype=quant_dtype,
        is_scale_transposed=False,
    )


# Bench functions
def bench_fn(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,
    label: str,
    sub_label: str,
    fn: Callable,
    description: str,
) -> TMeasurement:
    min_run_time = 1

    globals = {
        "x": x,
        "quant_dtype": quant_dtype,
        "group_size": group_size,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(x, quant_dtype, group_size)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench(params: bench_params_t, label: str, sub_label: str) -> Iterable[TMeasurement]:
    """Run benchmarks for all implementations."""
    # Make inputs: [num_tokens, hidden_size * 2] for [gate || up]
    scale = 1 / params.hidden_size
    x = (
        torch.randn(
            params.num_tokens,
            params.hidden_size * 2,
            dtype=params.dtype,
            device="cuda",
        )
        * scale
    )

    timers = []

    # Unfused per-tensor FP8
    timers.append(
        bench_fn(
            x,
            torch.float8_e4m3fn,
            params.group_size,
            label,
            sub_label,
            unfused_fp8_impl,
            "unfused_fp8_impl",
        )
    )

    # Unfused group-wise FP8
    timers.append(
        bench_fn(
            x,
            torch.float8_e4m3fn,
            params.group_size,
            label,
            sub_label,
            unfused_groupwise_fp8_impl,
            "unfused_groupwise_fp8_impl",
        )
    )

    # Fused group-wise FP8
    timers.append(
        bench_fn(
            x,
            torch.float8_e4m3fn,
            params.group_size,
            label,
            sub_label,
            fused_impl,
            "fused_groupwise_fp8_impl",
        )
    )

    # Triton group-wise FP8
    timers.append(
        bench_fn(
            x,
            torch.float8_e4m3fn,
            params.group_size,
            label,
            sub_label,
            triton_impl,
            "triton_groupwise_fp8_impl",
        )
    )

    print_timers(timers)

    return timers


def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def main():
    torch.set_default_device("cuda")
    bench_params = get_bench_params()

    print(f"Running {len(bench_params)} benchmark configurations...")
    print(f"This will take approximately {len(bench_params) * 3} seconds (1s per variant)")
    print()

    timers = []
    for bp in tqdm(bench_params):
        result_timers = bench(bp, "silu-mul-block-quant", bp.description())
        timers.extend(result_timers)
    
    print("\n" + "="*80)
    print("FINAL COMPARISON - ALL RESULTS")
    print("="*80)
    print_timers(timers)

    # Pickle all the results
    timestamp = int(time.time())
    filename = f"silu_mul_block_quant-{timestamp}.pkl"
    with open(filename, "wb") as f:
        pkl.dump(timers, f)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
