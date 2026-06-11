# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark comparing NVFP4 SiLU+Mul+Quant kernel variants:
  baseline   — vectorized BF16->FP4 (global loads)
  persistent — TMA warp-specialized persistent (3D TMA descriptors)

Methodology:
  - CUDA graphs with ArgPool rotation to defeat L2 cache
  - torch.utils.benchmark.Timer with blocked_autorange for statistical rigor

Usage:
    python benchmarks/kernels/benchmark_silu_mul_nvfp4_quant_tma_persistent.py
"""

import argparse

import torch
import torch.utils.benchmark as TBenchmark
import vllm._moe_C  # noqa: F401

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 persistent kernels require SM100+.")


def compute_sf_bytes(N: int, H: int) -> int:
    num_m_tiles = (N + 127) // 128
    num_k_tiles = (H + 63) // 64
    return num_m_tiles * num_k_tiles * 512


def make_nvfp4_inputs(N: int, H: int):
    input_bf16 = torch.randn(N, 2 * H, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    return input_bf16, global_scale


def make_nvfp4_outputs(N: int, H: int):
    sf_bytes = compute_sf_bytes(N, H)
    output = torch.empty(N, H // 2, dtype=torch.uint8, device="cuda")
    output_sf = torch.zeros(sf_bytes, dtype=torch.uint8, device="cuda")
    return output, output_sf


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

    print(f"NVFP4 SiLU+Mul+Quant benchmark: H={H}, nc={nc}, silu={silu_type}")
    print(f"HBM peak: {peak} TB/s")
    print(f"Methodology: {mode}, ArgPool={pool}, blocked_autorange")
    print()

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
        sf_bytes = compute_sf_bytes(N, H)
        total_nbytes = N * 2 * H * 2 + N * H // 2 + sf_bytes

        baseline_args_list = []
        for _ in range(pool):
            inp, gs = make_nvfp4_inputs(N, H)
            out, sf = make_nvfp4_outputs(N, H)
            mask = torch.tensor([N], dtype=torch.int32, device="cuda")
            baseline_args_list.append((out, sf, inp, gs, mask))

        def baseline_fn(out, sf, inp, gs, mask):
            torch.ops._moe_C.nvfp4_silu_mul_quant(out, sf, inp, gs, mask, 1)

        timer, n_ops = bench_fn(
            baseline_fn,
            baseline_args_list,
            "nvfp4-silu-mul-quant",
            f"N={N}",
            "baseline",
        )
        dt_base = timer.median / n_ops
        all_timers.append(timer)

        results = [("baseline", dt_base)]

        for bs in args.batch_sizes:
            persist_args_list = []
            for _ in range(pool):
                inp, gs = make_nvfp4_inputs(N, H)
                out, sf = make_nvfp4_outputs(N, H)
                n_tok = torch.tensor([N], dtype=torch.int32, device="cuda")
                persist_args_list.append(
                    (inp, out, sf, gs, n_tok, nc, bs, args.tanh_silu)
                )

            def persist_fn(inp, out, sf, gs, n_tok, nc_, bs_, tanh):
                torch.ops._moe_C.silu_mul_nvfp4_quant_tma_ws_persistent_bf16(
                    inp, out, sf, gs, n_tok, nc_, bs_, tanh
                )

            timer, n_ops = bench_fn(
                persist_fn,
                persist_args_list,
                "nvfp4-silu-mul-quant",
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
                f"{tbps:>7.2f} {pct:>5.1f}% {ratio:>7.02f}x{marker}"
            )
        print("-" * len(hdr))

    print()
    compare = TBenchmark.Compare(all_timers)
    compare.print()


if __name__ == "__main__":
    main()
