# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: Fused BMM+FP8 quant kernels vs separate BMM then FP8 quant.

Measures the latency of the MLA _v_up_proj operation for decode:
  - Static per-tensor FP8:
    - Baseline: torch.bmm + transpose + scaled_fp8_quant (3 ops)
    - Triton:   bmm_fp8_quant (single Triton kernel)
    - Helion:   bmm_fp8_quant_helion (if available)
  - Dynamic per-group FP8 (group_size=V=128):
    - Baseline: torch.bmm + transpose + per_token_group_fp8_quant
    - Triton:   bmm_fp8_group_quant (single Triton kernel)
    - Helion:   bmm_fp8_group_quant_helion (if available)

Usage:
    python benchmarks/kernels/benchmark_bmm_fp8_quant.py
"""

import torch

from vllm._custom_ops import scaled_fp8_quant
from vllm.kernels.triton.ops.bmm_fp8_quant import (
    bmm_fp8_group_quant,
    bmm_fp8_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton as vllm_triton
from vllm.utils.import_utils import has_helion

# DeepSeek-V3/R1 MLA dimensions
N_HEADS = [16, 64, 128]
KV_LORA_RANK = 512
V_HEAD_DIM = 128

# Detect Helion availability
_helion_available = False
_helion_group_available = False
if has_helion():
    try:
        from vllm.kernels.helion.ops.bmm_fp8_quant import bmm_fp8_quant_helion

        _helion_available = True
    except (ImportError, ValueError, RuntimeError):
        pass
    try:
        from vllm.kernels.helion.ops.bmm_fp8_quant import (
            bmm_fp8_group_quant_helion,
        )

        _helion_group_available = True
    except (ImportError, ValueError, RuntimeError):
        pass

# Build provider lists for static per-tensor benchmark
_line_vals = ["baseline", "triton"]
_line_names = ["torch.bmm + fp8_quant", "fused triton"]
_styles = [("blue", "-"), ("green", "-")]

if _helion_available:
    _line_vals.append("helion")
    _line_names.append("fused helion")
    _styles.append(("red", "-"))

# Build provider lists for per-group benchmark
_group_line_vals = ["group_baseline", "group_triton"]
_group_line_names = ["torch.bmm + group_fp8_quant", "fused group triton"]
_group_styles = [("blue", "--"), ("green", "--")]

if _helion_group_available:
    _group_line_vals.append("group_helion")
    _group_line_names.append("fused group helion")
    _group_styles.append(("red", "--"))


def _make_static_benchmark(n_heads):
    """Benchmark for static per-tensor FP8 quantization."""

    @vllm_triton.testing.perf_report(
        vllm_triton.testing.Benchmark(
            x_names=["B"],
            x_vals=[1, 4, 8, 16, 32, 64, 128, 256],
            line_arg="provider",
            line_vals=_line_vals,
            line_names=_line_names,
            styles=_styles,
            ylabel="Time (ms)",
            plot_name="bmm_fp8_static_N{}_L{}_V{}".format(
                n_heads, KV_LORA_RANK, V_HEAD_DIM
            ),
            args={
                "N": n_heads,
                "L": KV_LORA_RANK,
                "V": V_HEAD_DIM,
            },
        )
    )
    def benchmark(B, N, L, V, provider):
        device = "cuda"
        dtype = torch.bfloat16
        fp8_dtype = current_platform.fp8_dtype()

        inp = torch.randn(N, B, L, dtype=dtype, device=device)
        weight = torch.randn(N, L, V, dtype=dtype, device=device)
        scale = torch.tensor([0.01], dtype=torch.float32, device=device)

        out_bf16 = torch.empty(B, N * V, dtype=dtype, device=device)
        out_fp8 = torch.empty(B, N * V, dtype=fp8_dtype, device=device)

        if provider == "baseline":

            def fn():
                bmm_result = torch.bmm(inp, weight)
                out_bf16[:] = bmm_result.transpose(0, 1).reshape(B, N * V)
                out_fp8[:], _ = scaled_fp8_quant(out_bf16, scale)

        elif provider == "triton":

            def fn():
                bmm_fp8_quant(inp, weight, scale, out_fp8)

        elif provider == "helion":

            def fn():
                bmm_fp8_quant_helion(inp, weight, scale)

        ms = vllm_triton.testing.do_bench_cudagraph(fn, rep=500)
        return ms

    return benchmark


def _make_group_benchmark(n_heads):
    """Benchmark for dynamic per-group FP8 quantization."""
    _, fp8_max = get_fp8_min_max()
    fp8_info = torch.finfo(current_platform.fp8_dtype())

    @vllm_triton.testing.perf_report(
        vllm_triton.testing.Benchmark(
            x_names=["B"],
            x_vals=[1, 4, 8, 16, 32, 64, 128, 256],
            line_arg="provider",
            line_vals=_group_line_vals,
            line_names=_group_line_names,
            styles=_group_styles,
            ylabel="Time (ms)",
            plot_name="bmm_fp8_group_N{}_L{}_V{}".format(
                n_heads, KV_LORA_RANK, V_HEAD_DIM
            ),
            args={
                "N": n_heads,
                "L": KV_LORA_RANK,
                "V": V_HEAD_DIM,
            },
        )
    )
    def benchmark(B, N, L, V, provider):
        device = "cuda"
        dtype = torch.bfloat16
        fp8_dtype = current_platform.fp8_dtype()

        inp = torch.randn(N, B, L, dtype=dtype, device=device)
        weight = torch.randn(N, L, V, dtype=dtype, device=device)

        out_bf16 = torch.empty(B, N * V, dtype=dtype, device=device)
        out_fp8 = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
        out_scales = torch.empty(B, N, dtype=torch.float32, device=device)

        if provider == "group_baseline":

            def fn():
                bmm_result = torch.bmm(inp, weight)
                out_bf16[:] = bmm_result.transpose(0, 1).reshape(B, N * V)
                torch.ops._C.per_token_group_fp8_quant(
                    out_bf16,
                    out_fp8,
                    out_scales,
                    V,
                    1e-12,
                    fp8_info.min,
                    fp8_info.max,
                    False,
                    False,
                    False,
                )

        elif provider == "group_triton":

            def fn():
                bmm_fp8_group_quant(inp, weight, out_fp8, out_scales)

        elif provider == "group_helion":

            def fn():
                bmm_fp8_group_quant_helion(inp, weight)

        ms = vllm_triton.testing.do_bench_cudagraph(fn, rep=500)
        return ms

    return benchmark


if __name__ == "__main__":
    for n in N_HEADS:
        print(f"\n{'=' * 60}")
        print("  Static per-tensor FP8")
        print(f"  N_HEADS={n}, KV_LORA_RANK={KV_LORA_RANK}, V_HEAD_DIM={V_HEAD_DIM}")
        print(f"{'=' * 60}")
        bench = _make_static_benchmark(n)
        bench.run(print_data=True)

    print(f"\n\n{'#' * 60}")
    print(f"  Per-group dynamic FP8 (group_size={V_HEAD_DIM})")
    print(f"{'#' * 60}")

    for n in N_HEADS:
        print(f"\n{'=' * 60}")
        print("  Per-group FP8")
        print(f"  N_HEADS={n}, KV_LORA_RANK={KV_LORA_RANK}, V_HEAD_DIM={V_HEAD_DIM}")
        print(f"{'=' * 60}")
        bench = _make_group_benchmark(n)
        bench.run(print_data=True)
