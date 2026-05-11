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


def _fi_precondition(col_major: bool, gs: int) -> bool:
    """FI SM90 quantize requires column-major scales and group_size=128."""
    return col_major and gs == 128


# Column widths for consistent output
_FMT_CUDA = 10
_FMT_FI = 10
_FMT_SPEEDUP = 9
_FMT_TRITON = 10
_FMT_VS_TRITON = 9


def _run_single(
    shape: tuple[int, int],
    group_size: int,
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

    def impl():
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

    triton_ms = _time_cuda(triton_impl, warmup_iters, bench_iters)

    use_fi = _fi_precondition(column_major, group_size)

    if use_fi:
        # CUDA-only: mock FI to force fallback to CUDA kernel
        with patch.object(
            fp8_utils,
            "_flashinfer_sm90_per_token_group_quant_fp8",
            return_value=None,
        ):
            cuda_ms = _time_cuda(impl, warmup_iters, bench_iters)
        fi_ms = _time_cuda(impl, warmup_iters, bench_iters)
        fi_speedup = cuda_ms / fi_ms if fi_ms else math.inf
    else:
        cuda_ms = _time_cuda(impl, warmup_iters, bench_iters)
        fi_ms = None
        fi_speedup = None

    vs_triton = triton_ms / cuda_ms if cuda_ms else math.inf

    shape_s = f"({num_tokens}, {hidden_dim})"
    cfg_s = (
        f"{shape_s:>14}  gs={group_size:<3}  "
        f"col_major={int(column_major)}  ue8m0={int(scale_ue8m0)}"
    )

    cu_s = f"{cuda_ms:7.3f} ms"
    if fi_ms is not None:
        fi_s = f"{fi_ms:7.3f} ms"
        fi_sp = f"x{fi_speedup:5.2f}"
    else:
        fi_s = "      n/a"
        fi_sp = "    n/a"
    tr_s = f"{triton_ms:7.3f} ms"
    vt_s = f"x{vs_triton:5.2f}"

    print(
        f"{cfg_s:52} | {cu_s:>{_FMT_CUDA}} | {fi_s:>{_FMT_FI}} | "
        f"{fi_sp:>{_FMT_SPEEDUP}} | {tr_s:>{_FMT_TRITON}} | {vt_s:>{_FMT_VS_TRITON}}"
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

    header = (
        f"{'Configuration':52} | {'CUDA':>{_FMT_CUDA}} | {'FI/SM90':>{_FMT_FI}} | "
        f"{'FI spup':>{_FMT_SPEEDUP}} | {'Triton':>{_FMT_TRITON}} | {'vs Triton':>{_FMT_VS_TRITON}}"
    )
    print(header)
    print("-" * len(header))

    group_sizes = [128]
    for hidden_dim in hidden_dims:
        for num_tokens in num_tokens_list:
            shape = (num_tokens, hidden_dim)
            for gs in group_sizes:
                for col_major in (False, True):
                    for ue8m0 in (False, True):
                        _run_single(
                            shape,
                            gs,
                            column_major=col_major,
                            scale_ue8m0=ue8m0,
                            warmup_iters=warmup_iters,
                            bench_iters=bench_iters,
                        )
