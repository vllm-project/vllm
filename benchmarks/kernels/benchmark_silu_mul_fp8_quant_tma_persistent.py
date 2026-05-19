# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark comparing FP8 SiLU+Mul+FP8Quant kernel variants:
  baseline   — flashinfer vectorized (4 warps/CTA, global loads)
  persistent — TMA warp-specialized persistent (3D TMA descriptors)

Methodology:
  - CUDA graphs with ArgPool rotation to defeat L2 cache
  - torch.utils.benchmark.Timer with blocked_autorange for statistical rigor

Usage:
    python benchmarks/kernels/benchmark_silu_mul_fp8_quant_tma_persistent.py
    python benchmarks/kernels/benchmark_silu_mul_fp8_quant_tma_persistent.py \
        --batch-sizes 1 2 4 8 --tokens 16384 65536 262144
"""

import argparse

import torch
import torch.utils.benchmark as TBenchmark
import vllm._moe_C  # noqa: F401

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

FP8_DTYPE = current_platform.fp8_dtype()
E4M3_MAX = 448.0
GROUP_SIZE = 128


def make_fp8_input(N: int, H: int):
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
        [
            gate_fp8.reshape(N, H).to(FP8_DTYPE),
            up_fp8.reshape(N, H).to(FP8_DTYPE),
        ],
        dim=1,
    )

    input_scales = torch.empty(2 * G, N, dtype=torch.float32, device="cuda")
    input_scales[:G] = gate_scales.t()
    input_scales[G:] = up_scales.t()

    return input_fp8, input_scales


def make_fp8_outputs(N: int, H: int):
    G = H // GROUP_SIZE
    output = torch.empty(N, H, dtype=FP8_DTYPE, device="cuda")
    output_scales = torch.empty(G, N, dtype=torch.float32, device="cuda")
    return output, output_scales


def bench_cuda_graph(
    fn,
    args_list: list[tuple],
    label: str,
    sub_label: str,
    description: str,
) -> tuple[TBenchmark.Measurement, int]:
    """Benchmark with CUDA graph capture + ArgPool rotation."""
    n_pool = len(args_list)

    for args in args_list:
        fn(*args)
    torch.accelerator.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for args in args_list:
                fn(*args)

    timer = TBenchmark.Timer(
        stmt="g.replay()",
        globals={"g": g},
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=1)

    del g
    return timer, n_pool


def bench_eager(
    fn,
    args_list: list[tuple],
    label: str,
    sub_label: str,
    description: str,
) -> tuple[TBenchmark.Measurement, int]:
    """Benchmark with eager ArgPool rotation (fallback)."""
    n_args = len(args_list)

    for args in args_list:
        fn(*args)
    torch.accelerator.synchronize()

    idx = [0]

    def run_fn():
        fn(*args_list[idx[0] % n_args])
        idx[0] += 1

    timer = TBenchmark.Timer(
        stmt="run_fn()",
        globals={"run_fn": run_fn},
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=1)

    return timer, 1


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
    parser.add_argument(
        "--arg-pool-size",
        type=int,
        default=8,
        help="Number of input sets for L2 cache rotation",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Use eager timing instead of CUDA graphs",
    )
    args = parser.parse_args()

    set_random_seed(42)

    H = args.H
    nc = args.n_compute
    pool = args.arg_pool_size
    silu_type = "tanh" if args.tanh_silu else "real"
    peak = args.hbm_peak
    bench_fn = bench_eager if args.no_cuda_graph else bench_cuda_graph
    mode = "eager" if args.no_cuda_graph else "CUDA graphs"

    print(f"FP8 SiLU+Mul+FP8Quant benchmark: H={H}, nc={nc}, silu={silu_type}")
    print(f"HBM peak: {peak} TB/s")
    print(f"Methodology: {mode}, ArgPool={pool}, blocked_autorange")
    print()

    G = H // GROUP_SIZE

    hdr = (
        f"{'Kernel':<30} {'Tokens':>7} {'Time(us)':>10} "
        f"{'TB/s':>7} {'%peak':>6} {'vs base':>8}"
    )
    sep = "=" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    all_timers = []

    for N in args.tokens:
        total_nbytes = N * 2 * H + 2 * G * N * 4 + N * H + G * N * 4

        baseline_args_list = []
        for _ in range(pool):
            inp, scales = make_fp8_input(N, H)
            out, oscales = make_fp8_outputs(N, H)
            n_tok = torch.tensor([N], dtype=torch.int32, device="cuda")
            baseline_args_list.append(
                (inp, scales, out, oscales, n_tok, args.tanh_silu)
            )

        def baseline_fn(inp, scales, out, oscales, n_tok, tanh):
            torch.ops._moe_C.silu_mul_fp8_quant_baseline(
                inp, scales, out, oscales, n_tok, tanh
            )

        timer, n_ops = bench_fn(
            baseline_fn,
            baseline_args_list,
            "fp8-silu-mul-quant",
            f"N={N}",
            "baseline",
        )
        dt_base = timer.median / n_ops
        all_timers.append(timer)

        results = [("baseline", dt_base)]

        for bs in args.batch_sizes:
            persist_args_list = []
            for _ in range(pool):
                inp, scales = make_fp8_input(N, H)
                out, oscales = make_fp8_outputs(N, H)
                n_tok = torch.tensor([N], dtype=torch.int32, device="cuda")
                persist_args_list.append(
                    (inp, scales, out, oscales, n_tok, nc, bs, args.tanh_silu)
                )

            def persist_fn(inp, scales, out, oscales, n_tok, nc_, bs_, tanh):
                torch.ops._moe_C.silu_mul_fp8_quant_tma_ws_persistent(
                    inp, scales, out, oscales, n_tok, nc_, bs_, tanh
                )

            timer, n_ops = bench_fn(
                persist_fn,
                persist_args_list,
                "fp8-silu-mul-quant",
                f"N={N}",
                f"persist-nc{nc}-bs{bs}",
            )
            dt = timer.median / n_ops
            all_timers.append(timer)
            results.append((f"persist bs={bs}", dt))

        best_dt = min(r[1] for r in results)
        for name, dt in results:
            tbps = total_nbytes / dt / 1e12
            pct = tbps / peak * 100
            ratio = dt_base / dt
            marker = " <--" if dt == best_dt and dt < dt_base else ""
            print(
                f"{name:<30} {N:>7} {dt * 1e6:>10.1f} "
                f"{tbps:>7.2f} {pct:>5.1f}% {ratio:>7.2f}x{marker}"
            )
        print("-" * len(hdr))

    print()
    compare = TBenchmark.Compare(all_timers)
    compare.print()


if __name__ == "__main__":
    main()
