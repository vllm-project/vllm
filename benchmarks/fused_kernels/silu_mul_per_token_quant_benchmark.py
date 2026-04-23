# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Microbenchmark: Fused SiLU+Mul + per-token dynamic FP8 quantization.

Compares:
  - unfused_fp8_impl: SiLU+Mul (PyTorch) + per-tensor FP8 quant (C++)
  - unfused_per_token_fp8_impl: SiLU+Mul (C++) + per-token FP8 quant (C++)
  - fused_per_token_fp8_impl: Fused Triton kernel

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/fused_kernels/silu_mul_per_token_quant_benchmark.py
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from tqdm import tqdm

import vllm._custom_ops as ops

# Load the Triton kernel directly from source file since it may not be
# in the installed wheel yet.
import importlib.util as _ilu
import os as _os

_src = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    "vllm", "model_executor", "layers", "quantization", "triton_quantization.py",
)
try:
    from vllm.triton_utils import tl, triton  # noqa: F401
except ImportError:
    import sys
    import triton
    import triton.language as tl
    sys.modules["vllm.triton_utils"] = type(sys)("vllm.triton_utils")
    sys.modules["vllm.triton_utils"].tl = tl
    sys.modules["vllm.triton_utils"].triton = triton
_spec = _ilu.spec_from_file_location("_triton_quant", _src)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fused_silu_mul_per_token_quant = _mod.fused_silu_mul_per_token_quant


@dataclass
class bench_params_t:
    num_tokens: int
    hidden_size: int
    dtype: torch.dtype

    def description(self):
        return (
            f"N {self.num_tokens} "
            f"x D {self.hidden_size} "
            f"x DT {self.dtype}"
        )


def get_bench_params() -> list[bench_params_t]:
    """Test configurations covering common model sizes."""
    NUM_TOKENS = [16, 128, 512, 2048]
    HIDDEN_SIZES = [1024, 2048, 4096, 5120, 14336]  # Common FFN sizes
    DTYPES = [torch.float16, torch.bfloat16]

    combinations = product(NUM_TOKENS, HIDDEN_SIZES, DTYPES)
    bench_params = list(
        map(lambda x: bench_params_t(x[0], x[1], x[2]), combinations)
    )
    return bench_params


# Reference implementations
def unfused_fp8_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
):
    """Unfused: SiLU+Mul (PyTorch) then per-tensor quantize (C++)."""
    hidden = x.shape[-1] // 2
    gate, up = x.split(hidden, dim=-1)
    silu_out = F.silu(gate) * up
    silu_out, _ = ops.scaled_fp8_quant(silu_out)


def unfused_per_token_fp8_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
):
    """Unfused: SiLU+Mul (C++) then per-token dynamic FP8 quant (C++)."""
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    silu_out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(silu_out, x)
    fp8_out = torch.empty(output_shape, dtype=quant_dtype, device=x.device)
    scales = torch.empty(
        (x.shape[0], 1), dtype=torch.float32, device=x.device
    )
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(
        fp8_out, silu_out, scales, None
    )


def fused_per_token_fp8_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
):
    """Fused: SiLU+Mul + per-token FP8 quant in single Triton kernel."""
    fused_silu_mul_per_token_quant(x, torch.float8_e4m3fn)


# Bench functions
def bench_fn(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    label: str,
    sub_label: str,
    fn: Callable,
    description: str,
) -> TMeasurement:
    min_run_time = 1

    globals = {
        "x": x,
        "quant_dtype": quant_dtype,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(x, quant_dtype)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench(
    params: bench_params_t, label: str, sub_label: str
) -> Iterable[TMeasurement]:
    """Run benchmarks for all implementations."""
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
            label,
            sub_label,
            unfused_fp8_impl,
            "unfused_fp8_impl",
        )
    )

    # Unfused per-token FP8 (C++ silu_and_mul + C++ per_token_quant)
    timers.append(
        bench_fn(
            x,
            torch.float8_e4m3fn,
            label,
            sub_label,
            unfused_per_token_fp8_impl,
            "unfused_per_token_fp8_impl",
        )
    )

    # Fused per-token FP8 (Triton)
    timers.append(
        bench_fn(
            x,
            torch.float8_e4m3fn,
            label,
            sub_label,
            fused_per_token_fp8_impl,
            "fused_per_token_fp8_impl",
        )
    )

    return timers


def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def main():
    torch.set_default_device("cuda")
    bench_params = get_bench_params()

    print(f"Running {len(bench_params)} benchmark configurations...")
    print(
        f"This will take approximately {len(bench_params) * 3} seconds "
        f"(1s per variant)"
    )
    print()

    # Warmup: trigger Triton JIT compilation for all unique hidden sizes
    # so the first timed config doesn't include compilation overhead.
    print("Warming up Triton kernels...")
    for h in sorted({bp.hidden_size for bp in bench_params}):
        x_warm = torch.randn(16, h * 2, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            fused_silu_mul_per_token_quant(x_warm, torch.float8_e4m3fn)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    timers = []
    for bp in tqdm(bench_params):
        result_timers = bench(bp, "silu-mul-per-token-quant", bp.description())
        timers.extend(result_timers)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON - ALL RESULTS")
    print("=" * 80)
    print_timers(timers)


if __name__ == "__main__":
    main()
