# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark for FP8 TMA warp-specialized persistent SiLU+Mul+FP8Quant kernel.

Measures effective memory bandwidth (TB/s) and % of HBM peak across
different token counts, n_compute values, and batch sizes.

Usage:
    python -m benchmarks.kernels.benchmark_silu_mul_fp8_quant_tma_persistent
"""

import argparse

import torch
import vllm._moe_C  # noqa: F401 — loads the _moe_C extension

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


def benchmark_kernel(
    N: int,
    H: int,
    n_compute: int,
    batch_size: int,
    use_tanh_silu: bool,
    warmup: int = 10,
    iters: int = 100,
) -> float:
    """Returns elapsed time in milliseconds."""
    G = H // GROUP_SIZE
    input_fp8, input_scales = make_fp8_input(N, H)
    output = torch.empty(N, H, dtype=FP8_DTYPE, device="cuda")
    output_scales = torch.empty(G, N, dtype=torch.float32, device="cuda")
    n_tokens = torch.tensor([N], dtype=torch.int32, device="cuda")

    for _ in range(warmup):
        torch.ops._moe_C.silu_mul_fp8_quant_tma_ws_persistent(
            input_fp8,
            input_scales,
            output,
            output_scales,
            n_tokens,
            n_compute,
            batch_size,
            use_tanh_silu,
        )
    torch.accelerator.synchronize()

    start = torch.accelerator.Event(enable_timing=True)
    end = torch.accelerator.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.ops._moe_C.silu_mul_fp8_quant_tma_ws_persistent(
            input_fp8,
            input_scales,
            output,
            output_scales,
            n_tokens,
            n_compute,
            batch_size,
            use_tanh_silu,
        )
    end.record()
    torch.accelerator.synchronize()

    return start.elapsed_time(end) / iters


def compute_bandwidth(N: int, H: int, time_ms: float) -> float:
    """Compute effective bandwidth in TB/s."""
    G = H // GROUP_SIZE
    input_bytes = N * 2 * H * 1
    input_scale_bytes = N * 2 * G * 4
    output_bytes = N * H * 1
    output_scale_bytes = N * G * 4
    total_bytes = input_bytes + input_scale_bytes + output_bytes + output_scale_bytes
    return total_bytes / (time_ms * 1e-3) / 1e12


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

    print(f"FP8 persistent TMA benchmark: H={H}, nc={nc}, silu={silu_type}")
    print(f"HBM peak: {args.hbm_peak} TB/s")
    print()

    header = f"{'tokens':>8s}"
    for bs in args.batch_sizes:
        header += f"  {'bs=' + str(bs):>10s}"
    print(header)
    print("-" * len(header))

    for N in args.tokens:
        row = f"{N:>8d}"
        for bs in args.batch_sizes:
            t = benchmark_kernel(N, H, nc, bs, args.tanh_silu)
            bw = compute_bandwidth(N, H, t)
            pct = 100.0 * bw / args.hbm_peak
            row += f"  {bw:5.2f}/{pct:4.1f}%"
        print(row)


if __name__ == "__main__":
    main()
