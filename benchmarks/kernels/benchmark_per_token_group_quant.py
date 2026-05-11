# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import math
from collections.abc import Callable
from contextlib import contextmanager
from unittest.mock import patch

import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils, int8_utils
from vllm.platforms import current_platform


@contextmanager
def _triton_mode():
    """Temporarily force the Triton fallback path"""
    with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
        yield


def _time_cuda(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    warmup_iters: int,
    bench_iters: int,
) -> float:
    # warmup
    for _ in range(warmup_iters):
        fn()
    torch.accelerator.synchronize()

    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    start.record()
    for _ in range(bench_iters):
        fn()
    end.record()
    torch.accelerator.synchronize()

    return start.elapsed_time(end) / bench_iters  # ms/iter


def _can_run_fi(column_major: bool, group_size: int, scale_ue8m0: bool) -> bool:
    """FI SM90 path only works with column-major, TMA-aligned, group_size=128, no ue8m0."""
    return column_major and group_size == 128 and not scale_ue8m0


def _run_single(
    shape: tuple[int, int],
    group_size: int,
    dtype: str,
    *,
    column_major: bool = False,
    scale_ue8m0: bool = False,
    warmup_iters: int,
    bench_iters: int,
) -> None:
    num_tokens, hidden_dim = shape

    device = torch.device("cuda")
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16) * 8

    if dtype == "fp8":

        def cuda_impl():
            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size,
                column_major_scales=column_major,
                tma_aligned_scales=column_major,
                use_ue8m0=scale_ue8m0,
            )

        def triton_impl():
            with _triton_mode():
                return fp8_utils.per_token_group_quant_fp8(
                    x,
                    group_size,
                    column_major_scales=column_major,
                    tma_aligned_scales=column_major,
                    use_ue8m0=scale_ue8m0,
                )

        def fi_impl():
            """FI SM90 + CUDA fallback (same as cuda_impl but measured separately)."""
            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size,
                column_major_scales=column_major,
                tma_aligned_scales=column_major,
                use_ue8m0=scale_ue8m0,
            )

    elif dtype == "int8":

        def cuda_impl():
            return int8_utils.per_token_group_quant_int8(x, group_size)

        def triton_impl():
            with _triton_mode():
                return int8_utils.per_token_group_quant_int8(x, group_size)
    else:
        raise ValueError("dtype must be 'fp8' or 'int8'")

    cuda_ms = _time_cuda(cuda_impl, warmup_iters, bench_iters)
    triton_ms = _time_cuda(triton_impl, warmup_iters, bench_iters)
    speedup = triton_ms / cuda_ms if cuda_ms else math.inf

    cfg_desc = (
        f"shape={shape}  gs={group_size:<3}  col_major={column_major:<5}  "
        f"ue8m0={scale_ue8m0:<5}  dtype={dtype}"
    )

    if dtype == "fp8" and _can_run_fi(column_major, group_size, scale_ue8m0):
        # FI path: mock out _flashinfer_sm90_per_token_group_quant_fp8 to get
        # CUDA-only baseline, then compare against FI-enabled path.
        with patch.object(
            fp8_utils,
            "_flashinfer_sm90_per_token_group_quant_fp8",
            return_value=None,
        ):
            cuda_only_ms = _time_cuda(fi_impl, warmup_iters, bench_iters)
        fi_ms = _time_cuda(fi_impl, warmup_iters, bench_iters)
        fi_speedup = cuda_only_ms / fi_ms if fi_ms else math.inf
        print(
            f"{cfg_desc:55} | CUDA {cuda_only_ms:7.3f} ms  | FI    {fi_ms:7.3f} ms  | "
            f"FI speed-up ×{fi_speedup:5.2f}  | Triton {triton_ms:7.3f} ms  | "
            f"vs Triton ×{speedup:5.2f}"
        )
    else:
        print(
            f"{cfg_desc:55} | CUDA {cuda_ms:7.3f} ms  | Triton {triton_ms:7.3f} ms  | "
            f"speed-up ×{speedup:5.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--dtype", choices=["fp8", "int8", "both"], default="fp8")
    parser.add_argument(
        "--num-tokens", type=str, default="1,16,128,1024,4096",
        help="Comma-separated token counts",
    )
    parser.add_argument(
        "--hidden-dim", type=str, default="3072,6144",
        help="Comma-separated hidden dimensions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not current_platform.is_cuda():
        raise RuntimeError("CUDA device is required to run this benchmark.")

    args = parse_args()
    warmup_iters, bench_iters = args.warmup_iters, args.bench_iters

    num_tokens_list = [int(n) for n in args.num_tokens.split(",")]
    hidden_dims = [int(d) for d in args.hidden_dim.split(",")]

    dtypes = ["fp8", "int8"] if args.dtype == "both" else [args.dtype]

    header = (
        "Configuration".ljust(55)
        + " | "
        + "CUDA (ms)".center(12)
        + " | "
        + "FI (ms)".center(12)
        + " | "
        + "FI Speed-up".center(13)
        + " | "
        + "Triton (ms)".center(13)
        + " | "
        + "vs Triton"
    )
    print(header)
    print("-" * len(header))

    group_sizes = [128]  # FI only supports gs=128
    for dtype in dtypes:
        for hidden_dim in hidden_dims:
            for num_tokens in num_tokens_list:
                shape = (num_tokens, hidden_dim)
                for gs in group_sizes:
                    if dtype == "fp8":
                        for col_major in (False, True):
                            for ue8m0 in (False, True):
                                _run_single(
                                    shape,
                                    gs,
                                    dtype,
                                    column_major=col_major,
                                    scale_ue8m0=ue8m0,
                                    warmup_iters=warmup_iters,
                                    bench_iters=bench_iters,
                                )
                    else:
                        _run_single(
                            shape,
                            gs,
                            dtype,
                            warmup_iters=warmup_iters,
                            bench_iters=bench_iters,
                        )
