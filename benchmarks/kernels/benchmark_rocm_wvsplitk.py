# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare ROCm wvSplitK with the unquantized GEMM fallbacks on Kimi-K3."""

import argparse
import math
import statistics
from collections.abc import Callable
from functools import partial

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.utils.platform_utils import num_compute_units

KIMI_K3_TP8_SHAPES = (
    (163840, 256),
    (20480, 7168),
    (8448, 7168),
    (7168, 7168),
    (7168, 1536),
    (1536, 7168),
    (3584, 7168),
    (2112, 7168),
)
TOKEN_COUNTS = range(1, 6)
BOUNDARY_SHAPES = (
    *((m, 7168, (4, 5)) for m in (7168, 8191, 8192, 8193, 8448)),
    *((m, 256, TOKEN_COUNTS) for m in (20480, 32768, 65536, 98304, 131072, 163840)),
)
LLMM1_BOUNDARY_SHAPES = tuple(
    (m, 256, (1,))
    for m in (32768, 40960, 49152, 57344, 61440, 65536, 98304, 131072, 163840)
)


def _device_times_us(
    operations: tuple[tuple[str, Callable[[], object]], ...],
    iters: int,
    repeats: int,
) -> dict[str, float]:
    if iters < 1 or repeats < 1:
        raise ValueError("iters and repeats must be positive")

    for warmup in range(20):
        ordered = operations if warmup % 2 == 0 else operations[::-1]
        for _, fn in ordered:
            fn()
    torch.accelerator.synchronize()

    samples: dict[str, list[float]] = {name: [] for name, _ in operations}
    for repeat in range(repeats):
        ordered = operations if repeat % 2 == 0 else operations[::-1]
        for name, fn in ordered:
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.accelerator.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / iters)
    return {name: statistics.median(values) for name, values in samples.items()}


@torch.inference_mode()
def main(iters: int, repeats: int, shape_set: str) -> None:
    if iters < 1 or repeats < 1:
        raise ValueError("iters and repeats must be positive")
    if not torch.accelerator.is_available() or not current_platform.is_rocm():
        raise RuntimeError("this benchmark requires a ROCm GPU")

    try:
        from aiter.tuned_gemm import tgemm
    except ImportError:
        tgemm = None

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    if not on_gfx950():
        raise RuntimeError("this benchmark requires gfx950")
    cu_count = num_compute_units(device.index or 0)
    print(
        f"# device={current_platform.get_device_name()} gfx=gfx950 "
        f"cu_count={cu_count} dtype={dtype} "
        f"iters={iters} repeats={repeats}"
    )
    print(
        f"{'M':>7} {'K':>6} {'N':>2} {'wvSplitK':>10} {'LLMM1':>10} "
        f"{'tgemm':>10} {'torch':>10} {'ll/wv':>7} {'tg/wv':>7} "
        f"{'pt/wv':>7} {'max_err':>9}"
    )
    print(
        "# speedup columns are fallback latency / wvSplitK latency; >1 favors wvSplitK"
    )

    if shape_set == "kimi-k3":
        shapes = ((m, k, TOKEN_COUNTS) for m, k in KIMI_K3_TP8_SHAPES)
    elif shape_set == "llmm1-boundary":
        shapes = iter(LLMM1_BOUNDARY_SHAPES)
    else:
        shapes = iter(BOUNDARY_SHAPES)

    for m, k, token_counts in shapes:
        scale = math.sqrt(2 / k)
        weight = torch.randn(m, k, dtype=dtype, device=device) * scale
        for n in token_counts:
            x = torch.randn(n, k, dtype=dtype, device=device) * scale
            ref = torch.nn.functional.linear(x, weight)
            actual = ops.wvSplitK(weight, x, cu_count, None)
            atol = torch.finfo(dtype).eps * math.sqrt(k)
            torch.testing.assert_close(actual, ref, atol=atol, rtol=1e-2)
            max_error = (actual - ref).abs().max().item()

            operations: list[tuple[str, Callable[[], object]]] = [
                ("wvsplitk", partial(ops.wvSplitK, weight, x, cu_count, None)),
                ("torch", partial(torch.nn.functional.linear, x, weight)),
            ]
            if tgemm is not None:
                aiter_actual = tgemm.mm(x, weight, None)
                torch.testing.assert_close(aiter_actual, ref, atol=atol, rtol=1e-2)
                operations.append(("tgemm", partial(tgemm.mm, x, weight, None)))
            if n == 1 and m % 4 == 0 and k <= 8192:
                llmm1_actual = ops.LLMM1(weight, x, 4)
                torch.testing.assert_close(llmm1_actual, ref, atol=atol, rtol=1e-2)
                operations.append(("llmm1", partial(ops.LLMM1, weight, x, 4)))

            timings = _device_times_us(tuple(operations), iters, repeats)
            wvsplitk = timings["wvsplitk"]
            torch_gemm = timings["torch"]
            aiter_gemm = timings.get("tgemm", float("nan"))
            llmm1 = timings.get("llmm1", float("nan"))
            print(
                f"{m:>7} {k:>6} {n:>2} {wvsplitk:>10.2f} {llmm1:>10.2f} "
                f"{aiter_gemm:>10.2f} {torch_gemm:>10.2f} "
                f"{llmm1 / wvsplitk:>7.2f} {aiter_gemm / wvsplitk:>7.2f} "
                f"{torch_gemm / wvsplitk:>7.2f} {max_error:>9.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--shape-set",
        choices=("kimi-k3", "boundary", "llmm1-boundary"),
        default="kimi-k3",
    )
    args = parser.parse_args()
    main(args.iters, args.repeats, args.shape_set)
