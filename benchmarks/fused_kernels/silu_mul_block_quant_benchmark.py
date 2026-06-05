# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
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
    use_ue8m0: bool,
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
    use_ue8m0: bool,
):
    """Unfused: SiLU+Mul then group-wise quantize."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)

    # SiLU(gate) * up
    silu_out = F.silu(gate) * up

    # Group quantize - use group_size directly
    silu_out, _ = per_token_group_quant_fp8(
        silu_out, group_size=group_size, use_ue8m0=use_ue8m0
    )


def fused_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    group_size: int,
    use_ue8m0: bool,
):
    """Fused: SiLU+Mul+Block Quantization in single kernel."""
    out, _ = ops.silu_and_mul_per_block_quant(
        x,
        group_size=group_size,
        quant_dtype=quant_dtype,
        is_scale_transposed=False,
        use_ue8m0=use_ue8m0,
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
    use_ue8m0: bool,
) -> TMeasurement:
    min_run_time = 1

    globals = {
        "x": x,
        "quant_dtype": quant_dtype,
        "group_size": group_size,
        "use_ue8m0": use_ue8m0,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(x, quant_dtype, group_size, use_ue8m0)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench(
    params: bench_params_t,
    label: str,
    sub_label: str,
    use_ue8m0: bool,
) -> Iterable[TMeasurement]:
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

    if not use_ue8m0:
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
                use_ue8m0,
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
            "unfused_groupwise_fp8_impl"
            if not use_ue8m0
            else "unfused_groupwise_fp8_ue8m0_impl",
            use_ue8m0,
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
            "fused_groupwise_fp8_impl"
            if not use_ue8m0
            else "fused_groupwise_fp8_ue8m0_impl",
            use_ue8m0,
        )
    )

    return timers


def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-ue8m0",
        action="store_true",
        help="Benchmark UE8M0-rounded FP32 scales for groupwise FP8 paths.",
    )
    args = parser.parse_args()

    torch.set_default_device("cuda")
    bench_params = get_bench_params()
    variants_per_config = 2 if args.use_ue8m0 else 3
    label = (
        "silu-mul-block-quant-ue8m0"
        if args.use_ue8m0
        else "silu-mul-block-quant"
    )

    print(f"Running {len(bench_params)} benchmark configurations...")
    print(
        "This will take approximately "
        f"{len(bench_params) * variants_per_config} seconds (1s per variant)"
    )
    print(f"UE8M0 scales: {args.use_ue8m0}")
    print()

    timers = []
    for bp in tqdm(bench_params):
        result_timers = bench(bp, label, bp.description(), args.use_ue8m0)
        timers.extend(result_timers)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON - ALL RESULTS")
    print("=" * 80)
    print_timers(timers)


if __name__ == "__main__":
    main()
