# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark comparing NVFP4 SiLU+Mul+Quant kernel variants:
  baseline   — vectorized BF16->FP4 (global loads)
  persistent — TMA warp-specialized persistent (3D TMA descriptors)

Usage:
    python benchmarks/kernels/benchmark_silu_mul_nvfp4_quant_tma_persistent.py
"""

import argparse
import gc

import torch
import vllm._moe_C  # noqa: F401

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 persistent kernels require SM100+.")


def compute_sf_bytes(N: int, H: int) -> int:
    num_m_tiles = (N + 127) // 128
    num_k_tiles = (H + 63) // 64
    return num_m_tiles * num_k_tiles * 512


def free_gpu():
    gc.collect()
    torch.accelerator.empty_cache()


def cuda_event_bench(fn, warmup=10, iters=100):
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


def bench_baseline(N: int, H: int) -> tuple[float, int]:
    input_bf16 = torch.randn(N, 2 * H, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    output = torch.empty(N, H // 2, dtype=torch.uint8, device="cuda")
    sf_bytes = compute_sf_bytes(N, H)
    output_sf = torch.zeros(sf_bytes, dtype=torch.uint8, device="cuda")
    mask = torch.tensor([N], dtype=torch.int32, device="cuda")

    dt = cuda_event_bench(
        lambda: torch.ops._moe_C.nvfp4_silu_mul_quant(
            output, output_sf, input_bf16, global_scale, mask, 1
        )
    )

    total_nbytes = N * 2 * H * 2 + N * H // 2 + sf_bytes
    return dt, total_nbytes


def bench_persistent(
    N: int, H: int, n_compute: int, batch_size: int, use_tanh_silu: bool
) -> tuple[float, int]:
    input_bf16 = torch.randn(N, 2 * H, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    output = torch.empty(N, H // 2, dtype=torch.uint8, device="cuda")
    sf_bytes = compute_sf_bytes(N, H)
    output_sf = torch.zeros(sf_bytes, dtype=torch.uint8, device="cuda")
    n_tokens = torch.tensor([N], dtype=torch.int32, device="cuda")

    dt = cuda_event_bench(
        lambda: torch.ops._moe_C.silu_mul_nvfp4_quant_tma_ws_persistent_bf16(
            input_bf16,
            output,
            output_sf,
            global_scale,
            n_tokens,
            n_compute,
            batch_size,
            use_tanh_silu,
        )
    )

    total_nbytes = N * 2 * H * 2 + N * H // 2 + sf_bytes
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

    print(f"NVFP4 SiLU+Mul+Quant benchmark: H={H}, nc={nc}, silu={silu_type}")
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
        dt_base, nbytes = bench_baseline(N, H)
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
