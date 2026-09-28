# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: Fused FP8 output quantization in merge_attn_states

Compares fused vs unfused approaches for producing FP8-quantized merged
attention output:
  1. Fused CUDA     -- single CUDA kernel (merge + FP8 quant)
  2. Fused Triton   -- single Triton kernel (merge + FP8 quant)
  3. Unfused CUDA   -- CUDA merge + torch.compiled FP8 quant
  4. Unfused Triton  -- Triton merge + torch.compiled FP8 quant

Usage:
    python benchmarks/fused_kernels/merge_attn_states_benchmarks.py
    python benchmarks/fused_kernels/merge_attn_states_benchmarks.py --tp 1 4 8
    python benchmarks/fused_kernels/merge_attn_states_benchmarks.py --dtype bfloat16
"""

import argparse
import itertools

import torch

from vllm._custom_ops import merge_attn_states as merge_attn_states_cuda
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_merge_attn_states import (
    merge_attn_states as merge_attn_states_triton,
)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

NUM_TOKENS_LIST = [1, 16, 64, 256, 1024, 4096]

# (label, num_heads, head_size) — num_heads is for TP=1
HEAD_CONFIGS = [
    ("DeepSeek-V3 MLA", 128, 128),
    ("Llama-70B", 64, 128),
    ("Llama-8B", 32, 128),
]

TP_SIZES = [1, 2, 4, 8]

INPUT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

QUANTILES = [0.5, 0.2, 0.8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def short_dtype(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def make_inputs(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    """Create random prefix/suffix outputs and LSEs."""
    prefix_output = torch.randn(
        (num_tokens, num_heads, head_size), dtype=dtype, device="cuda"
    )
    suffix_output = torch.randn(
        (num_tokens, num_heads, head_size), dtype=dtype, device="cuda"
    )
    prefix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device="cuda")
    suffix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device="cuda")
    # Sprinkle some inf values to exercise edge-case paths
    mask = torch.rand(num_heads, num_tokens, device="cuda") < 0.05
    prefix_lse[mask] = float("inf")
    mask2 = torch.rand(num_heads, num_tokens, device="cuda") < 0.05
    suffix_lse[mask2] = float("inf")
    return prefix_output, suffix_output, prefix_lse, suffix_lse


def build_configs(head_configs, num_tokens_list, input_dtypes, tp_sizes):
    """Build (num_tokens, num_heads, head_size, dtype_str) config tuples,
    applying TP division to num_heads and skipping invalid combos."""
    configs = []
    for (_, nh, hs), nt, dtype, tp in itertools.product(
        head_configs, num_tokens_list, input_dtypes, tp_sizes
    ):
        nh_tp = nh // tp
        if nh_tp >= 1:
            configs.append((nt, nh_tp, hs, short_dtype(dtype)))
    return configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark merge_attn_states fused FP8 quantization"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=None,
        help=f"Override token counts (default: {NUM_TOKENS_LIST})",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="+",
        default=None,
        help=f"TP sizes to simulate (divides num_heads) (default: {TP_SIZES})",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="+",
        default=None,
        help="Input dtypes (e.g. bfloat16 float16 float32). "
        f"Default: {[short_dtype(d) for d in INPUT_DTYPES]}",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parse args and build configs before decorators
# ---------------------------------------------------------------------------

args = parse_args()

num_tokens_list = args.num_tokens if args.num_tokens else NUM_TOKENS_LIST
tp_sizes = args.tp if args.tp else TP_SIZES

if args.dtype:
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

    input_dtypes = [STR_DTYPE_TO_TORCH_DTYPE[d] for d in args.dtype]
else:
    input_dtypes = INPUT_DTYPES

configs = build_configs(HEAD_CONFIGS, num_tokens_list, input_dtypes, tp_sizes)

torch._dynamo.config.recompile_limit = 8888


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_heads", "head_size", "dtype_str"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["fused_cuda", "fused_triton", "unfused_cuda", "unfused_triton"],
        line_names=["Fused CUDA", "Fused Triton", "Unfused CUDA", "Unfused Triton"],
        styles=[("blue", "-"), ("green", "-"), ("blue", "--"), ("green", "--")],
        ylabel="us",
        plot_name="merge_attn_states FP8 (fused vs unfused)",
        args={},
    )
)
@default_vllm_config()
def benchmark(num_tokens, num_heads, head_size, dtype_str, provider):
    input_dtype = getattr(torch, dtype_str)
    fp8_dtype = current_platform.fp8_dtype()
    prefix_out, suffix_out, prefix_lse, suffix_lse = make_inputs(
        num_tokens, num_heads, head_size, input_dtype
    )
    output_scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")

    if provider == "fused_cuda":
        output = torch.empty(
            (num_tokens, num_heads, head_size), dtype=fp8_dtype, device="cuda"
        )
        fn = lambda: merge_attn_states_cuda(
            output,
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
            output_scale=output_scale,
        )
    elif provider == "fused_triton":
        output = torch.empty(
            (num_tokens, num_heads, head_size), dtype=fp8_dtype, device="cuda"
        )
        fn = lambda: merge_attn_states_triton(
            output,
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
            output_scale=output_scale,
        )
    elif provider == "unfused_cuda":
        merge_buf = torch.empty(
            (num_tokens, num_heads, head_size), dtype=input_dtype, device="cuda"
        )
        quant_fp8 = QuantFP8(
            static=True,
            group_shape=GroupShape.PER_TENSOR,
            column_major_scales=False,
        )
        quant_input = merge_buf.view(-1, head_size)
        compiled_quant = torch.compile(
            quant_fp8.forward_native, fullgraph=True, dynamic=False
        )

        def unfused_fn():
            merge_attn_states_cuda(
                merge_buf, prefix_out, prefix_lse, suffix_out, suffix_lse
            )
            compiled_quant(quant_input, output_scale)

        fn = unfused_fn
    else:  # unfused_triton
        merge_buf = torch.empty(
            (num_tokens, num_heads, head_size), dtype=input_dtype, device="cuda"
        )
        quant_fp8 = QuantFP8(
            static=True,
            group_shape=GroupShape.PER_TENSOR,
            column_major_scales=False,
        )
        quant_input = merge_buf.view(-1, head_size)
        compiled_quant = torch.compile(
            quant_fp8.forward_native, fullgraph=True, dynamic=False
        )

        def unfused_fn():
            merge_attn_states_triton(
                merge_buf, prefix_out, prefix_lse, suffix_out, suffix_lse
            )
            compiled_quant(quant_input, output_scale)

        fn = unfused_fn

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=QUANTILES)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms  # us


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device_name = current_platform.get_device_name()
    print(f"Device: {device_name}")
    print(f"Token counts: {num_tokens_list}")
    print(f"TP sizes: {tp_sizes}")
    print(f"Input dtypes: {[short_dtype(d) for d in input_dtypes]}")
    print(f"Head configs: {[(c[0], c[1], c[2]) for c in HEAD_CONFIGS]}")
    benchmark.run(print_data=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
