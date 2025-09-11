# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import torch

from vllm.platforms import current_platform
from vllm.utils import cdiv
from vllm.utils.deep_gemm import per_block_cast_to_fp8 as per_block_cast_to_fp8_triton


def _align(x: int, y: int) -> int:
    return cdiv(x, y) * y


@torch.compile
def per_block_cast_to_fp8_baseline(
    x: torch.Tensor,
    block_size: list[int],
    use_ue8m0: bool,
):
    assert x.dim() == 2
    m, n = x.shape
    block_m, block_n = block_size

    m_pad, n_pad = _align(m, block_m), _align(n, block_n)
    x_padded = torch.zeros((m_pad, n_pad), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x

    # [num_blocks_m, block_m, num_blocks_n, block_n]
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)

    # per-block amax -> scale
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    if use_ue8m0:
        # round up to nearest power-of-two (UE8M0)
        sf = torch.pow(2.0, torch.ceil(torch.log2(sf.abs())))

    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    y = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales_2d = sf.view(x_view.size(0), x_view.size(2))
    return y, scales_2d


def _time_it(fn, warmup: int, iters: int):
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return sum(times_ms) / len(times_ms)


def run_case(
    m: int, n: int, dtype: torch.dtype, block_m: int, block_n: int, use_ue8m0: bool
) -> tuple[float, float, bool, float, bool, float]:
    x = torch.randn((m, n), dtype=dtype, device="cuda")
    block_size = [block_m, block_n]

    # pre-run to compile Triton
    y_new, s_new = per_block_cast_to_fp8_triton(
        x, block_size=block_size, use_ue8m0=use_ue8m0
    )
    y_ref, s_ref = per_block_cast_to_fp8_baseline(
        x, block_size=block_size, use_ue8m0=use_ue8m0
    )
    torch.cuda.synchronize()

    # correctness checks
    y_new, y_ref = y_new.float(), y_ref.float()
    same_y = torch.allclose(y_new, y_ref, rtol=1e-6, atol=0.0)
    max_y_diff = float((y_new - y_ref).abs().max().item())
    same_s = torch.allclose(s_new, s_ref, rtol=1e-6, atol=0.0)
    max_s_diff = float((s_new - s_ref).abs().max().item())

    # baseline timing
    t_ref = _time_it(
        lambda: per_block_cast_to_fp8_baseline(
            x, block_size=block_size, use_ue8m0=use_ue8m0
        ),
        warmup=5,
        iters=50,
    )

    # triton timing
    t_new = _time_it(
        lambda: per_block_cast_to_fp8_triton(
            x, block_size=block_size, use_ue8m0=use_ue8m0
        ),
        warmup=5,
        iters=50,
    )

    return t_ref, t_new, same_y, max_y_diff, same_s, max_s_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--use-ue8m0", action="store_true")
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    args = parser.parse_args()
    current_platform.seed_everything(42)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shapes = [
        (128, 128),
        (1024, 1024),
        (2048, 4096),
        (4096, 4096),
        (4096, 8192),
        (8192, 4096),
        (3000, 4097),  # non-multiple
        (7168, 7168),  # common transformer dim
        (16384, 32768),  # large shape
    ]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"dtype={args.dtype}, UE8M0={args.use_ue8m0}, "
        f"block=({args.block_m}x{args.block_n})"
    )
    print(
        f"{'Shape':>14} | {'Baseline (ms)':>12} | {'Triton (ms)':>11} | "
        f"{'Speedup':>7} | {'Y equal':>7} | {'Y maxdiff':>9} | "
        f"{'S equal':>7} | {'S maxdiff':>9}"
    )
    print("-" * 86)

    for m, n in shapes:
        t_ref, t_new, same_y, max_y_diff, same_s, max_s_diff = run_case(
            m, n, dtype, args.block_m, args.block_n, args.use_ue8m0
        )
        speedup = t_ref / t_new if t_new > 0 else float("inf")
        shape_str = f"{m}x{n}"
        print(
            f"{shape_str:>14} | {t_ref:12.3f} | {t_new:11.3f} | {speedup:7.2f} "
            f"| {str(same_y):>7} | {max_y_diff:9.2e} | "
            f"{str(same_s):>7} | {max_s_diff:9.2e}"
        )
