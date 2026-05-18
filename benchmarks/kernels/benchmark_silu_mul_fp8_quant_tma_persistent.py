# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark comparing FP8 SiLU+Mul+FP8Quant kernel variants:
  baseline  — flashinfer vectorized (4 warps/CTA, global loads)
  persistent — TMA warp-specialized persistent (3D TMA descriptors)

Usage:
    python -m benchmarks.kernels.benchmark_silu_mul_fp8_quant_tma_persistent
"""

import argparse
import gc

import torch
import vllm._moe_C  # noqa: F401

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

FP8_DTYPE = current_platform.fp8_dtype()
E4M3_MAX = 448.0
GROUP_SIZE = 128


def make_fp8_input(N: int, H: int):
    """Create random FP8 input with per-group scales."""
    G = H // GROUP_SIZE
    bf16 = torch.randn(N, 2 * H, dtype=torch.bfloat16, device="cuda")

    gate = bf16[:, :H].float().reshape(N, G, GROUP_SIZE)
    up = bf16[:, H:].float().reshape(N, G, GROUP_SIZE)

    gate_amax = gate.abs().amax(dim=-1).clamp(min=1e-12)
    up_amax = up.abs().amax(dim=-1).clamp(min=1e-12)
    gate_scales = gate_amax / E4M3_MAX
    up_scales = up_amax / E4M3_MAX

    gate_fp8 = (gate / gate_scales.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
    up_fp8 = (up / up_scales.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)

    input_fp8 = torch.cat(
        [gate_fp8.reshape(N, H).to(FP8_DTYPE), up_fp8.reshape(N, H).to(FP8_DTYPE)],
        dim=1,
    )

    input_scales = torch.empty(2 * G, N, dtype=torch.float32, device="cuda")
    input_scales[:G] = gate_scales.t()
    input_scales[G:] = up_scales.t()

    return input_fp8, input_scales


def free_gpu():
    gc.collect()
    torch.accelerator.empty_cache()


def cuda_event_bench(fn, warmup=10, iters=100):
    """Benchmark using CUDA events, returns time in seconds."""
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = torch.Event("cuda", enable_timing=True)
    end = torch.Event("cuda", enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def bench_baseline(N: int, H: int, use_tanh_silu: bool) -> tuple[float, int]:
    """Benchmark baseline kernel. Returns (time_s, total_nbytes)."""
    G = H // GROUP_SIZE
    input_fp8, input_scales = make_fp8_input(N, H)
    output = torch.empty(N, H, dtype=FP8_DTYPE, device="cuda")
    output_scales = torch.empty(G, N, dtype=torch.float32, device="cuda")
    n_tokens = torch.tensor([N], dtype=torch.int32, device="cuda")

    # Baseline expects flat scale layout matching input_scales
    dt = cuda_event_bench(
        lambda: torch.ops._moe_C.silu_mul_fp8_quant_baseline(
            input_fp8, input_scales, output, output_scales, n_tokens, use_tanh_silu
        )
    )

    total_nbytes = N * 2 * H + 2 * G * N * 4 + N * H + G * N * 4
    return dt, total_nbytes


def bench_persistent(
    N: int, H: int, n_compute: int, batch_size: int, use_tanh_silu: bool
) -> tuple[float, int]:
    """Benchmark persistent kernel. Returns (time_s, total_nbytes)."""
    G = H // GROUP_SIZE
    input_fp8, input_scales = make_fp8_input(N, H)
    output = torch.empty(N, H, dtype=FP8_DTYPE, device="cuda")
    output_scales = torch.empty(G, N, dtype=torch.float32, device="cuda")
    n_tokens = torch.tensor([N], dtype=torch.int32, device="cuda")

    dt = cuda_event_bench(
        lambda: torch.ops._moe_C.silu_mul_fp8_quant_tma_ws_persistent(
            input_fp8,
            input_scales,
            output,
            output_scales,
            n_tokens,
            n_compute,
            batch_size,
            use_tanh_silu,
        )
    )

    total_nbytes = N * 2 * H + 2 * G * N * 4 + N * H + G * N * 4
    return dt, total_nbytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=7168)
    parser.add_argument("--n-compute", type=int, default=7)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1024, 4096, 16384, 65536, 262144],
    )
    parser.add_argument("--tanh-silu", action="store_true")
    parser.add_argument("--hbm-peak", type=float, default=8.0, help="HBM peak TB/s")
    args = parser.parse_args()

    set_random_seed(42)

    H = args.H
    nc = args.n_compute
    silu_type = "tanh" if args.tanh_silu else "real"
    peak = args.hbm_peak

    print(f"FP8 SiLU+Mul+FP8Quant benchmark: H={H}, nc={nc}, silu={silu_type}")
    print(f"HBM peak: {peak} TB/s")
    print()

    hdr = (
        f"{'Kernel':<30} {'Tokens':>7} {'Time(us)':>10} "
        f"{'TB/s':>7} {'%peak':>6} {'vs base':>8}"
    )
    sep = "=" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    for N in args.tokens:
        dt_base, nbytes = bench_baseline(N, H, args.tanh_silu)
        free_gpu()
        results = [("baseline", dt_base, nbytes)]

        for bs in args.batch_sizes:
            dt, nbytes = bench_persistent(N, H, nc, bs, args.tanh_silu)
            free_gpu()
            results.append((f"persist bs={bs}", dt, nbytes))

        best_dt = min(r[1] for r in results)
        for name, dt, nbytes in results:
            tbps = nbytes / dt / 1e12
            pct = tbps / peak * 100
            ratio = dt_base / dt
            marker = " <--" if dt == best_dt and dt < dt_base else ""
            print(
                f"{name:<30} {N:>7} {dt * 1e6:>10.1f} "
                f"{tbps:>7.2f} {pct:>5.1f}% {ratio:>7.2f}x{marker}"
            )
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
