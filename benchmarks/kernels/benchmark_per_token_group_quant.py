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
    elif dtype == "int8":

        def cuda_impl():
            return int8_utils.per_token_group_quant_int8(x, group_size)

        def triton_impl():
            with _triton_mode():
                return int8_utils.per_token_group_quant_int8(x, group_size)
    else:
        raise ValueError("dtype must be 'fp8' or 'int8'")

    triton_ms = _time_cuda(triton_impl, warmup_iters, bench_iters)

    # FI SM90 only applies to FP8 with col_major=True and group_size=128
    use_fi = dtype == "fp8" and column_major and group_size == 128
    if use_fi:
        with patch.object(
            fp8_utils,
            "_flashinfer_sm90_per_token_group_quant_fp8",
            return_value=None,
        ):
            cuda_ms = _time_cuda(cuda_impl, warmup_iters, bench_iters)
        fi_ms = _time_cuda(cuda_impl, warmup_iters, bench_iters)
        fi_speedup = cuda_ms / fi_ms if fi_ms else math.inf
    else:
        cuda_ms = _time_cuda(cuda_impl, warmup_iters, bench_iters)
        fi_ms = None
        fi_speedup = None

    vs_triton = triton_ms / cuda_ms if cuda_ms else math.inf

    cfg_desc = (
        f"shape={shape}  gs={group_size:<3}  col_major={column_major:<5}  "
        f"ue8m0={scale_ue8m0:<5}  dtype={dtype}"
    )

    cu_s = f"{cuda_ms:7.3f} ms"
    if fi_ms is not None:
        fi_s = f"{fi_ms:7.3f} ms"
        fi_sp = f"x{fi_speedup:5.2f}"
    else:
        fi_s = "     n/a"
        fi_sp = "   n/a"
    tr_s = f"{triton_ms:7.3f} ms"
    vt_s = f"x{vs_triton:5.2f}"

    print(
        f"{cfg_desc:55} | {cu_s:>10} | {fi_s:>10} | "
        f"{fi_sp:>8} | {tr_s:>10} | {vt_s:>8}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--dtype", choices=["fp8", "int8", "both"], default="both")
    return parser.parse_args()


if __name__ == "__main__":
    if not current_platform.is_cuda():
        raise RuntimeError("CUDA device is required to run this benchmark.")

    args = parse_args()
    warmup_iters, bench_iters = args.warmup_iters, args.bench_iters

    shapes = [
        (32, 128), (64, 256), (16, 512),
        (16, 3072), (128, 3072), (1024, 3072), (4096, 3072),
        (16, 6144), (128, 6144), (1024, 6144), (4096, 6144),
    ]
    group_sizes = [64, 128]

    dtypes = ["fp8", "int8"] if args.dtype == "both" else [args.dtype]

    header = (
        "Configuration".ljust(55)
        + " | "
        + "CUDA".center(10)
        + " | "
        + "FI/SM90".center(10)
        + " | "
        + "FI spup".center(8)
        + " | "
        + "Triton".center(10)
        + " | "
        + "vs Triton"
    )
    print(header)
    print("-" * len(header))

    for dtype in dtypes:
        for shape in shapes:
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
                else:  # INT8 has no col-major / ue8m0 switches
                    _run_single(
                        shape,
                        gs,
                        dtype,
                        warmup_iters=warmup_iters,
                        bench_iters=bench_iters,
                    )
