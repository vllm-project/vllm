# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark for the stride-aware FP8 QKV head-dim padding kernel.

Compares the original Triton implementation
(``quantize_fp8_pad_head_dim_triton``) against the new CUDA fast path
(``torch.ops._C.qkv_padded_fp8_quant``) on Qwen3-VL ViT-shaped workloads.

Run:
    # Default: scan all four (skip_scale x non_contig) modes
    python benchmarks/kernels/benchmark_qkv_padded_fp8.py

    # Verify CUDA bit-exactly matches Triton before benchmarking
    python benchmarks/kernels/benchmark_qkv_padded_fp8.py --check

    # Single-mode runs (legacy CLI flags still supported)
    python benchmarks/kernels/benchmark_qkv_padded_fp8.py --mode skip_scale
    python benchmarks/kernels/benchmark_qkv_padded_fp8.py --mode non_contig
"""

from __future__ import annotations

import torch

from vllm.kernels.triton.qkv_padded_fp8_quant import (
    quantize_fp8_pad_head_dim_triton,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

SHAPES: list[tuple[str, int, int, int]] = [
    # 1 image at the model's max num_position_embeddings (48x48 grid =
    # 1536x1536 px input).
    ("1img max-res (48x48)", 2304, 16, 72),
    # 4 medium-res images (~1024 tokens each, e.g. 1024x1024 px).
    ("4img mid-res", 4096, 16, 72),
    # 16 medium-res images / video frames -- representative multi-image
    # batch where the kernel becomes a noticeable fraction of attn time.
    ("16img / video", 16384, 16, 72),
    # Stress test: bandwidth-bound regime, exercises grid-stride
    # ROWS_PER_THREAD=4 and confirms the HBM3 ceiling (~2.6 TB/s on H20).
    ("64K tokens (stress)", 65536, 16, 72),
]

# Modes scanned by default: cartesian product of skip_scale x non_contig.
ALL_MODES: list[tuple[str, bool, bool]] = [
    ("default", False, False),
    ("skip_scale", True, False),
    ("non_contig", False, True),
    ("skip+nonctg", True, True),
]


def _make_input(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    non_contig: bool,
) -> torch.Tensor:
    """Build a benchmark input tensor.

    When ``non_contig`` is True we synthesize an interleaved QKV buffer
    of shape (S, 3H, D) and return the Q-slice ``qkv[:, 0::3, :]``,
    which has stride(-1)=1 but stride(-2)=3*D -- exactly the layout
    produced by a fused QKV projection in the ViT.
    """
    if non_contig:
        qkv = torch.randn(seq_len, 3 * num_heads, head_dim, device=device, dtype=dtype)
        x = qkv[:, 0::3, :]
        assert x.stride(-1) == 1
        return x
    return torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)


def _check_correctness(dtype: torch.dtype, device: torch.device) -> None:
    """Verify CUDA output bit-exactly matches the Triton reference.

    Checking up-front gives reviewers confidence that the perf numbers
    below come from semantically identical implementations.
    """
    print("[check] verifying CUDA output bit-exactly matches Triton ...")
    for name, S, H, D in SHAPES:
        for mode, skip_scale, non_contig in ALL_MODES:
            x = _make_input(S, H, D, dtype, device, non_contig)
            scale = torch.tensor([0.1], device=device, dtype=torch.float32)
            y_ref = quantize_fp8_pad_head_dim_triton(x, scale, skip_scale=skip_scale)
            y_cuda = torch.ops._C.qkv_padded_fp8_quant(x, scale, skip_scale)
            torch.testing.assert_close(
                y_cuda.float(),
                y_ref.float(),
                rtol=0.0,
                atol=0.0,
                msg=f"[{name} / {mode}] CUDA != Triton",
            )
    print("[check] OK -- all configurations bit-exactly match.")


def _bench_one(fn) -> tuple[float, float, float]:
    """Benchmark ``fn`` with CUDA graphs; returns (median, min, max) ms."""
    median, low, high = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5, 0.2, 0.8])
    return median, low, high


def _run_mode(
    mode: str,
    skip_scale: bool,
    non_contig: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Run all SHAPES under one (skip_scale, non_contig) mode."""
    print()
    print(f"=== mode: {mode}  (skip_scale={skip_scale}, non_contig={non_contig}) ===")
    print(
        f"{'Workload':<22}{'Triton ms':>12}{'CUDA ms':>12}"
        f"{'Speedup':>10}{'Triton GB/s':>14}{'CUDA GB/s':>12}"
    )
    print("-" * 82)

    for name, S, H, D in SHAPES:
        x = _make_input(S, H, D, dtype, device, non_contig)
        scale = torch.tensor([0.1], device=device, dtype=torch.float32)
        pad_D = (D + 15) // 16 * 16

        # Bytes-in counts only the unique elements actually read by the
        # kernel (S*H*D) -- not the full underlying QKV buffer in the
        # non-contig case, so the GB/s number reflects useful work.
        bytes_in = S * H * D * x.element_size()
        bytes_out = S * H * pad_D  # fp8 = 1 byte
        bytes_io = bytes_in + bytes_out

        triton_fn = lambda x=x, scale=scale: quantize_fp8_pad_head_dim_triton(
            x, scale, skip_scale=skip_scale
        )
        cuda_fn = lambda x=x, scale=scale: torch.ops._C.qkv_padded_fp8_quant(
            x, scale, skip_scale
        )

        t_triton, _, _ = _bench_one(triton_fn)
        t_cuda, _, _ = _bench_one(cuda_fn)

        bw_triton = bytes_io / (t_triton * 1e-3) / 1e9
        bw_cuda = bytes_io / (t_cuda * 1e-3) / 1e9
        print(
            f"{name:<22}{t_triton:>12.4f}{t_cuda:>12.4f}"
            f"{t_triton / t_cuda:>9.2f}x{bw_triton:>13.0f}{bw_cuda:>11.0f}"
        )


def main() -> None:
    parser = FlexibleArgumentParser(
        description=(
            "Benchmark torch.ops._C.qkv_padded_fp8_quant vs. the Triton "
            "reference on Qwen3-VL ViT-shaped workloads."
        )
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Input dtype (output is always FP8 e4m3fn).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "default", "skip_scale", "non_contig", "skip+nonctg"],
        default="all",
        help="Which (skip_scale, non_contig) combination to benchmark. "
        "'all' scans every combination (default).",
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="Before benchmarking, assert that the CUDA kernel produces "
        "bit-identical output to the Triton reference for every "
        "(shape, mode) pair.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if not hasattr(torch.ops._C, "qkv_padded_fp8_quant"):
        raise SystemExit(
            "torch.ops._C.qkv_padded_fp8_quant is not registered; "
            "rebuild vLLM with the new kernel."
        )

    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    device = torch.device("cuda")

    if args.check:
        _check_correctness(dtype, device)

    modes = (
        ALL_MODES if args.mode == "all" else [m for m in ALL_MODES if m[0] == args.mode]
    )
    for mode_name, skip_scale, non_contig in modes:
        _run_mode(mode_name, skip_scale, non_contig, dtype, device)


if __name__ == "__main__":
    main()
