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

Quant modes (selected via --quant-mode):
  * static    : per-tensor static FP8 (default)
  * group128  : per-token-per-group FP8 with group_size=128
  * group64   : per-token-per-group FP8 with group_size=64

Usage (script = benchmarks/fused_kernels/merge_attn_states_benchmarks.py):
    python <script>
    python <script> --tp 1 4 8
    python <script> --dtype bfloat16
    python <script> --quant-mode group128
    python <script> --quant-mode group128 --ue8m0
"""

import argparse
import itertools

import torch

from vllm._custom_ops import merge_attn_states as merge_attn_states_cuda
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
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

QUANT_MODES = ["static", "group128", "group64"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def short_dtype(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def quant_mode_group_size(mode: str) -> int | None:
    """Return the group size for a group-FP8 mode, or None for static."""
    if mode == "group128":
        return 128
    if mode == "group64":
        return 64
    if mode == "static":
        return None
    raise ValueError(f"unknown quant mode: {mode}")


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


def alloc_group_fp8_sf(
    num_tokens: int, num_heads: int, head_size: int, group_size: int
) -> torch.Tensor:
    """Allocate a row-major SF tensor compatible with per_token_group_quant_fp8."""
    num_groups = num_heads * head_size // group_size
    return torch.empty((num_tokens, num_groups), dtype=torch.float32, device="cuda")


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
    parser.add_argument(
        "--quant-mode",
        type=str,
        choices=QUANT_MODES,
        default="static",
        help="Output quant mode (default: static)",
    )
    parser.add_argument(
        "--ue8m0",
        action="store_true",
        help="Round per-group FP8 scales to power-of-2 (UE8M0). "
        "Only used when --quant-mode is group128 or group64.",
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

quant_mode: str = args.quant_mode
group_size: int | None = quant_mode_group_size(quant_mode)
ue8m0: bool = args.ue8m0

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
        plot_name=f"merge_attn_states FP8 ({quant_mode}, fused vs unfused)",
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

    # Skip cases where head_size isn't a multiple of group_size — the kernel
    # rejects these on the host side, so reporting them in the benchmark is
    # misleading.
    if group_size is not None and head_size % group_size != 0:
        return float("nan"), float("nan"), float("nan")

    if quant_mode == "static":
        fn = _build_static_fp8_fn(
            provider,
            num_tokens,
            num_heads,
            head_size,
            input_dtype,
            fp8_dtype,
            prefix_out,
            suffix_out,
            prefix_lse,
            suffix_lse,
        )
    else:
        assert group_size is not None
        fn = _build_group_fp8_fn(
            provider,
            num_tokens,
            num_heads,
            head_size,
            input_dtype,
            fp8_dtype,
            group_size,
            ue8m0,
            prefix_out,
            suffix_out,
            prefix_lse,
            suffix_lse,
        )

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=QUANTILES)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms  # us


def _build_static_fp8_fn(
    provider,
    num_tokens,
    num_heads,
    head_size,
    input_dtype,
    fp8_dtype,
    prefix_out,
    suffix_out,
    prefix_lse,
    suffix_lse,
):
    output_scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")

    if provider == "fused_cuda":
        output = torch.empty(
            (num_tokens, num_heads, head_size), dtype=fp8_dtype, device="cuda"
        )
        return lambda: merge_attn_states_cuda(
            output,
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
            output_scale=output_scale,
        )
    if provider == "fused_triton":
        output = torch.empty(
            (num_tokens, num_heads, head_size), dtype=fp8_dtype, device="cuda"
        )
        return lambda: merge_attn_states_triton(
            output,
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
            output_scale=output_scale,
        )

    # Unfused paths: merge → bf16, then a torch.compiled FP8 quant kernel.
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
    merge = (
        merge_attn_states_cuda
        if provider == "unfused_cuda"
        else merge_attn_states_triton
    )

    def unfused_fn():
        merge(merge_buf, prefix_out, prefix_lse, suffix_out, suffix_lse)
        compiled_quant(quant_input, output_scale)

    return unfused_fn


def _build_group_fp8_fn(
    provider,
    num_tokens,
    num_heads,
    head_size,
    input_dtype,
    fp8_dtype,
    group_size,
    ue8m0,
    prefix_out,
    suffix_out,
    prefix_lse,
    suffix_lse,
):
    if provider == "fused_cuda":
        output = torch.empty(
            (num_tokens, num_heads, head_size), dtype=fp8_dtype, device="cuda"
        )
        sf = alloc_group_fp8_sf(num_tokens, num_heads, head_size, group_size)
        return lambda: merge_attn_states_cuda(
            output,
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
            output_block_scale=sf,
            quant_group_size=group_size,
            quant_scale_ue8m0=ue8m0,
        )
    if provider == "fused_triton":
        output = torch.empty(
            (num_tokens, num_heads, head_size), dtype=fp8_dtype, device="cuda"
        )
        sf = alloc_group_fp8_sf(num_tokens, num_heads, head_size, group_size)
        return lambda: merge_attn_states_triton(
            output,
            prefix_out,
            prefix_lse,
            suffix_out,
            suffix_lse,
            output_block_scale=sf,
            quant_group_size=group_size,
            quant_scale_ue8m0=ue8m0,
        )

    # Unfused paths: merge → bf16, then per_token_group_quant_fp8.
    merge_buf = torch.empty(
        (num_tokens, num_heads, head_size), dtype=input_dtype, device="cuda"
    )
    quant_input = merge_buf.view(num_tokens, num_heads * head_size)
    quant_out = torch.empty_like(quant_input, dtype=fp8_dtype)
    merge = (
        merge_attn_states_cuda
        if provider == "unfused_cuda"
        else merge_attn_states_triton
    )

    def unfused_fn():
        merge(merge_buf, prefix_out, prefix_lse, suffix_out, suffix_lse)
        per_token_group_quant_fp8(
            quant_input,
            group_size,
            dtype=fp8_dtype,
            use_ue8m0=ue8m0,
            out_q=quant_out,
        )

    return unfused_fn


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
    print(f"Quant mode: {quant_mode}" + (" (UE8M0)" if ue8m0 else ""))
    benchmark.run(print_data=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
