"""
Benchmark: CuTe DSL kernel vs CUTLASS C++ kernel for per-tensor FP8 scaled GEMM
on NVIDIA Hopper (SM90).

Compares:
  1. CuTe DSL  - vllm/kernels/quantization/cutedsl/scaled_mm_sm90_fp8.py
                 (ScaledMmSm90Fp8Kernel)
  2. CUTLASS   - called via vllm._custom_ops.cutlass_scaled_mm
                 (csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh)

Both kernels compute:  D = scale_a * scale_b * (A @ B^T)
  - A: (M, K) FP8 e4m3, row-major
  - B: (N, K) FP8 e4m3, col-major  (stored as (N, K).T for CUTLASS C++)
  - D: (M, N) BF16 or FP16
  - scale_a, scale_b: per-tensor FP32 scalars

Usage:
    python scripts/benchmark_cutedsl_kernel_raw.py

    # Custom sizes
    python scripts/benchmark_cutedsl_kernel_raw.py \
        --m 128,256,512,1024 --k 4096 --n 4096

    # Use auto tile selection (mirrors C++ dispatch logic)
    python scripts/benchmark_cutedsl_kernel_raw.py \
        --auto-tile

    # Override CuTe DSL tile/cluster
    python scripts/benchmark_cutedsl_kernel_raw.py \
        --tile-shape-mn 128,128 --cluster-shape-mn 2,1
"""

import argparse
import os
import sys
import time
from typing import Callable, List, Tuple

import torch

# ---------------------------------------------------------------------------
# CuTe DSL imports (may not be installed everywhere)
# ---------------------------------------------------------------------------
_cute_dsl_available = False
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch

    _cute_dsl_available = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# CUTLASS via vLLM custom ops
# ---------------------------------------------------------------------------
_cutlass_ops_available = False
try:
    from vllm import _custom_ops as ops

    _cutlass_ops_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

CUTE_C_DTYPE_MAP = {}
if _cute_dsl_available:
    CUTE_C_DTYPE_MAP = {
        "bf16": cutlass.BFloat16,
        "fp16": cutlass.Float16,
    }


# ---------------------------------------------------------------------------
# Tensor creation: CUTLASS C++ path (PyTorch tensors)
# ---------------------------------------------------------------------------
def make_tensors_cutlass(
    m: int, n: int, k: int, out_dtype: torch.dtype
) -> Tuple[torch.Tensor, ...]:
    """Create FP8 A/B and per-tensor FP32 scales for the CUTLASS C++ kernel.

    CUTLASS expects:
      a: (M, K) fp8, row-major (contiguous)
      b: (N, K).T -> (K, N) fp8, column-major
      scale_a: scalar fp32 (per-tensor)
      scale_b: scalar fp32 (per-tensor)
    """
    a = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
    # B stored as (N, K) transposed to (K, N) column-major
    b = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn).t()

    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    out = torch.empty((m, n), device="cuda", dtype=out_dtype)
    return a, b, scale_a, scale_b, out


# ---------------------------------------------------------------------------
# Tensor creation: CuTe DSL path (CuTe tensors)
# ---------------------------------------------------------------------------
def make_tensors_cutedsl(m: int, n: int, k: int, c_dtype_str: str):
    """Create CuTe-style 3D tensors for the CuTe DSL kernel.

    A: (M, K, 1) K-major, B: (N, K, 1) K-major, C: (M, N, 1) N-major.
    scale_a: (M, 1, 1) per-token FP32 (filled with 1.0 for per-tensor).
    scale_b: (1, 1, 1) scalar FP32.
    """
    ab_dtype = cutlass.Float8E4M3FN
    c_dtype = CUTE_C_DTYPE_MAP[c_dtype_str]
    scale_dtype = cutlass.Float32
    l = 1

    torch.manual_seed(42)
    a_cpu = cutlass_torch.matrix(l, m, k, False, ab_dtype)
    b_cpu = cutlass_torch.matrix(l, n, k, False, ab_dtype)
    c_cpu = cutlass_torch.matrix(l, m, n, False, c_dtype)
    # scale_a is per-token (M, 1, 1) — uniform 1.0 for per-tensor benchmarking
    sfa_cpu = cutlass_torch.matrix(1, m, 1, False, scale_dtype)
    sfb_cpu = cutlass_torch.matrix(1, 1, 1, False, scale_dtype)

    a_tensor, _ = cutlass_torch.cute_tensor_like(
        a_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, _ = cutlass_torch.cute_tensor_like(
        b_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, _ = cutlass_torch.cute_tensor_like(
        c_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )
    sfa_tensor, _ = cutlass_torch.cute_tensor_like(
        sfa_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16
    )
    sfb_tensor, _ = cutlass_torch.cute_tensor_like(
        sfb_cpu, scale_dtype, is_dynamic_layout=True, assumed_align=16
    )

    return a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor


# ---------------------------------------------------------------------------
# CUDA-event based timing
# ---------------------------------------------------------------------------
def bench_cuda_events(
    fn: Callable, warmup: int = 10, iters: int = 100
) -> Tuple[float, float, float]:
    """Time *fn* with CUDA events. Returns (median_ms, mean_ms, min_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    minimum = times[0]
    return median, mean, minimum


# ---------------------------------------------------------------------------
# CuTe DSL kernel wrapper (compile once, launch many)
# ---------------------------------------------------------------------------
class CuteDslBenchWrapper:
    """Wraps the CuTe DSL ScaledMmSm90Fp8Kernel for benchmarking.

    Compiles the kernel once during __init__ and re-launches on each __call__.
    """

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        c_dtype_str: str,
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        # Ensure the kernel module is importable
        dsl_dir = os.path.join(
            os.path.dirname(__file__), "..", "vllm", "kernels", "quantization", "cutedsl"
        )
        dsl_dir = os.path.abspath(dsl_dir)
        if dsl_dir not in sys.path:
            sys.path.insert(0, dsl_dir)

        from scaled_mm_sm90_fp8 import ScaledMmSm90Fp8Kernel

        self.tensors = make_tensors_cutedsl(m, n, k, c_dtype_str)
        a_t, b_t, c_t, sfa_t, sfb_t = self.tensors

        self.gemm = ScaledMmSm90Fp8Kernel(
            acc_dtype=cutlass.Float32,
            tile_shape_mn=tile_shape_mn,
            cluster_shape_mn=cluster_shape_mn,
        )

        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

        # Determine compilation opt level
        try:
            from cutlass import CUDA_VERSION

            opt_level = (
                3
                if CUDA_VERSION.major < 13
                or (CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1)
                else 2
            )
        except ImportError:
            opt_level = 3

        # Compile once
        print(
            f"      [CuTe DSL] Compiling kernel "
            f"(tile={tile_shape_mn}, cluster={cluster_shape_mn}) ... ",
            end="",
            flush=True,
        )
        t0 = time.time()
        self.compiled = cute.compile(
            self.gemm,
            a_t,
            b_t,
            c_t,
            sfa_t,
            sfb_t,
            self.stream,
            options=f"--opt-level {opt_level}",
        )
        print(f"done ({time.time() - t0:.1f}s)")

    def __call__(self):
        a_t, b_t, c_t, sfa_t, sfb_t = self.tensors
        self.compiled(a_t, b_t, c_t, sfa_t, sfb_t, self.stream)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------
def run_benchmarks(
    m_sizes: List[int],
    k_sizes: List[int],
    n_sizes: List[int],
    out_dtype_str: str,
    warmup: int,
    iters: int,
    tile_shape_mn: Tuple[int, int] | None,
    cluster_shape_mn: Tuple[int, int] | None,
    auto_tile: bool,
):
    out_dtype = OUT_DTYPE_MAP[out_dtype_str]

    header = (
        f"{'M':>6}  {'N':>6}  {'K':>6}  "
        f"{'Kernel':<45}  "
        f"{'Median(ms)':>11}  {'Mean(ms)':>11}  {'Min(ms)':>10}  {'TFLOPS':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for m in m_sizes:
        for n in n_sizes:
            for k in k_sizes:
                flops = 2.0 * m * n * k
                results: List[Tuple[str, float, float, float, float]] = []

                # ---- CUTLASS C++ (via vLLM ops) ----
                if _cutlass_ops_available:
                    a, b, sa, sb, out = make_tensors_cutlass(m, n, k, out_dtype)

                    def cutlass_fn(_a=a, _b=b, _sa=sa, _sb=sb, _dt=out_dtype):
                        ops.cutlass_scaled_mm(_a, _b, _sa, _sb, _dt)

                    try:
                        med, mean, mn = bench_cuda_events(
                            cutlass_fn, warmup, iters
                        )
                        tflops = flops / (med * 1e-3) / 1e12
                        results.append(
                            ("CUTLASS C++ (vllm ops)", med, mean, mn, tflops)
                        )
                    except Exception as e:
                        results.append(
                            (f"CUTLASS C++ [ERR: {e}]", -1, -1, -1, 0)
                        )
                else:
                    results.append(
                        ("CUTLASS C++ [NOT AVAILABLE]", -1, -1, -1, 0)
                    )

                # ---- CuTe DSL ----
                if _cute_dsl_available:
                    # Resolve tile/cluster config
                    if auto_tile:
                        from scaled_mm_sm90_fp8 import select_tile_config

                        t_mn, c_mn = select_tile_config(m, n, k)
                    elif tile_shape_mn is not None and cluster_shape_mn is not None:
                        t_mn, c_mn = tile_shape_mn, cluster_shape_mn

                    else:
                        # Default
                        t_mn, c_mn = (128, 128), (2, 1)

                    try:
                        wrapper = CuteDslBenchWrapper(
                            m,
                            n,
                            k,
                            out_dtype_str,
                            tile_shape_mn=t_mn,
                            cluster_shape_mn=c_mn,
                        )
                        med, mean, mn = bench_cuda_events(
                            wrapper, warmup, iters
                        )
                        tflops = flops / (med * 1e-3) / 1e12
                        label = f"CuTe DSL (tile={t_mn}, cl={c_mn})"
                        results.append((label, med, mean, mn, tflops))
                    except Exception as e:
                        results.append(
                            (f"CuTe DSL [ERR: {e}]", -1, -1, -1, 0)
                        )
                else:
                    results.append(
                        ("CuTe DSL [NOT AVAILABLE]", -1, -1, -1, 0)
                    )

                # ---- PyTorch _scaled_mm baseline (optional) ----
                try:
                    a_pt = torch.randn((m, k), device="cuda").to(
                        torch.float8_e4m3fn
                    )
                    b_pt = torch.randn((n, k), device="cuda").to(
                        torch.float8_e4m3fn
                    ).t()
                    sa_pt = torch.tensor(1.0, device="cuda", dtype=torch.float32)
                    sb_pt = torch.tensor(1.0, device="cuda", dtype=torch.float32)

                    def pytorch_fn(
                        _a=a_pt, _b=b_pt, _sa=sa_pt, _sb=sb_pt, _dt=out_dtype
                    ):
                        torch._scaled_mm(
                            _a, _b, _sa, _sb, out_dtype=_dt, use_fast_accum=True
                        )

                    med, mean, mn = bench_cuda_events(
                        pytorch_fn, warmup, iters
                    )
                    tflops = flops / (med * 1e-3) / 1e12
                    results.append(
                        ("PyTorch _scaled_mm (fast_accum)", med, mean, mn, tflops)
                    )
                except Exception as e:
                    results.append(
                        (f"PyTorch _scaled_mm [ERR: {e}]", -1, -1, -1, 0)
                    )

                # Print results for this (M, N, K)
                for name, med, mean, mn, tflops in results:
                    if med < 0:
                        print(
                            f"{m:>6}  {n:>6}  {k:>6}  {name:<45}  "
                            f"{'N/A':>11}  {'N/A':>11}  {'N/A':>10}  {'N/A':>8}"
                        )
                    else:
                        print(
                            f"{m:>6}  {n:>6}  {k:>6}  {name:<45}  "
                            f"{med:>11.4f}  {mean:>11.4f}  {mn:>10.4f}  "
                            f"{tflops:>8.2f}"
                        )

                # Speedup summary
                cutlass_med = next(
                    (r[1] for r in results if "CUTLASS C++" in r[0] and r[1] > 0),
                    None,
                )
                cute_med = next(
                    (r[1] for r in results if "CuTe DSL" in r[0] and r[1] > 0),
                    None,
                )
                if cutlass_med and cute_med:
                    speedup = cutlass_med / cute_med
                    faster = "CuTe DSL" if speedup > 1 else "CUTLASS C++"
                    ratio = speedup if speedup > 1 else 1.0 / speedup
                    print(
                        f"{'':>6}  {'':>6}  {'':>6}  "
                        f"{'>> ' + faster + f' is {ratio:.2f}x faster':<45}"
                    )
                print()

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",")]


def parse_int_tuple(s: str) -> Tuple[int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated ints")
    return (parts[0], parts[1])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CuTe DSL vs CUTLASS C++ per-tensor FP8 scaled GEMM "
            "on NVIDIA Hopper (SM90)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--m",
        type=parse_int_list,
        default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        help="Comma-separated M dimensions (default: 128,256,...,8192)",
    )
    parser.add_argument(
        "--n",
        type=parse_int_list,
        default=[4096],
        help="Comma-separated N dimensions (default: 4096)",
    )
    parser.add_argument(
        "--k",
        type=parse_int_list,
        default=[4096],
        help="Comma-separated K dimensions (default: 4096)",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Output dtype (default: bf16)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of timed iterations (default: 100)",
    )
    parser.add_argument(
        "--auto-tile",
        action="store_true",
        default=True,
        help="Auto-select CuTe DSL tile/cluster using select_tile_config() "
        "(mirrors C++ dispatch, default: True)",
    )
    parser.add_argument(
        "--tile-shape-mn",
        type=parse_int_tuple,
        default=None,
        help="Override CuTe DSL tile shape M,N (e.g. 128,128). "
        "Disables --auto-tile.",
    )
    parser.add_argument(
        "--cluster-shape-mn",
        type=parse_int_tuple,
        default=None,
        help="Override CuTe DSL cluster shape M,N (e.g. 2,1). "
        "Disables --auto-tile.",
    )

    args = parser.parse_args()

    # If user explicitly sets tile/cluster, disable auto-tile
    if args.tile_shape_mn is not None or args.cluster_shape_mn is not None:
        args.auto_tile = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this benchmark.")

    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        raise RuntimeError(
            f"SM90+ (Hopper) GPU required, got sm_{cap[0]}{cap[1]}"
        )

    # Ensure CuTe DSL kernel module is importable
    if _cute_dsl_available:
        dsl_dir = os.path.join(
            os.path.dirname(__file__), "..", "vllm", "kernels", "quantization", "cutedsl"
        )
        dsl_dir = os.path.abspath(dsl_dir)
        if dsl_dir not in sys.path:
            sys.path.insert(0, dsl_dir)

    print()
    print("=" * 70)
    print("  Per-Tensor FP8 Scaled GEMM Benchmark: CuTe DSL vs CUTLASS C++")
    print("  Target: NVIDIA Hopper (SM90)")
    print("=" * 70)
    print(f"  GPU             : {torch.cuda.get_device_name()}")
    print(f"  Compute cap     : {cap[0]}.{cap[1]}")
    print(f"  M sizes         : {args.m}")
    print(f"  N sizes         : {args.n}")
    print(f"  K sizes         : {args.k}")
    print(f"  Output dtype    : {args.out_dtype}")
    print(f"  Warmup iters    : {args.warmup}")
    print(f"  Timed iters     : {args.iters}")
    print(f"  Auto tile select: {args.auto_tile}")
    if not args.auto_tile:
        print(f"  Tile shape MN   : {args.tile_shape_mn}")
        print(f"  Cluster shape MN: {args.cluster_shape_mn}")
    print(f"  CuTe DSL avail  : {_cute_dsl_available}")
    print(f"  CUTLASS ops avail: {_cutlass_ops_available}")
    print()

    run_benchmarks(
        m_sizes=args.m,
        k_sizes=args.k,
        n_sizes=args.n,
        out_dtype_str=args.out_dtype,
        warmup=args.warmup,
        iters=args.iters,
        tile_shape_mn=args.tile_shape_mn,
        cluster_shape_mn=args.cluster_shape_mn,
        auto_tile=args.auto_tile,
    )


if __name__ == "__main__":
    main()
