"""
Benchmark: SM103 (B300) FP4 Ultra GEMM vs SM100 (B200) NVFP4 GEMM
===================================================================

This benchmark compares the performance of the SM103-optimized FP4 Ultra
GEMM kernel against the SM100 NVFP4 GEMM kernel, both running on B300
hardware.  It also benchmarks the effect of Programmatic Dependent Launch
(PDL) on the quant->GEMM pipeline, where the GEMM consumer can begin
before the quant producer finishes.

SM103 kernels use:
  - K=768 tile (vs K=256 on SM100)
  - FP4 Ultra MMA (UltraVs16) schedule
  - NoSmemWarpSpecialized epilogue
  - Sm103BlockScaledConfig scale factor layout

PDL kernels additionally set:
  - cudaLaunchAttributeProgrammaticStreamSerialization on quant (producer)
  - CUTLASS launch_with_pdl=true on GEMM (enables overlap with next kernel)

Usage:
    python benchmarks/kernels/benchmark_nvfp4_sm103.py [--mode gemm|quant|e2e|pdl|all]

Requirements:
    - B300 GPU (SM103 / compute capability 10.3)
    - CUDA >= 12.9
    - vLLM built with ENABLE_NVFP4_SM100=1 and SM103 support
"""

import argparse
from typing import Optional

import torch
import vllm._C  # noqa: F401 - registers ops into torch.ops._C

# ============================================================================
# Helpers
# ============================================================================


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def get_sm_version() -> int:
    """Return SM version as integer (e.g., 100, 103, 120)."""
    cap = torch.cuda.get_device_capability()
    return cap[0] * 10 + cap[1]


def create_nvfp4_tensors(
    m: int, n: int, k: int, dtype: torch.dtype = torch.bfloat16
) -> dict:
    """
    Create synthetic NVFP4 GEMM input tensors (A, B, scales, alpha).

    A: [m, k/2] uint8 (packed FP4)
    B: [n, k/2] uint8 (packed FP4, column-major)
    A_sf: [round_up(m,128), round_up(k/16,4)] float8_e4m3fn (SM100 swizzled)
    B_sf: [round_up(n,128), round_up(k/16,4)] float8_e4m3fn (SM100 swizzled)
    alpha: [1] float32
    D: [m, n] output
    """
    # Packed FP4 data (random bytes -- content doesn't affect timing)
    A = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device="cuda")
    B = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")

    # Scale factors (SM100 swizzled layout)
    sf_m = round_up(m, 128)
    sf_n = round_up(n, 128)
    sf_k = round_up(k // 16, 4)

    A_sf_sm100 = torch.randint(
        0, 256, (sf_m, sf_k), dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e4m3fn)
    B_sf_sm100 = torch.randint(
        0, 256, (sf_n, sf_k), dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e4m3fn)

    # SM103 layout: convert from SM100 layout
    A_sf_sm103 = torch.empty_like(A_sf_sm100)
    B_sf_sm103 = torch.empty_like(B_sf_sm100)
    torch.ops._C.convert_sf_layout_sm100_to_sm103(A_sf_sm103, A_sf_sm100)
    torch.ops._C.convert_sf_layout_sm100_to_sm103(B_sf_sm103, B_sf_sm100)

    # Global alpha
    alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # Output
    D = torch.empty(m, n, dtype=dtype, device="cuda")

    return {
        "A": A,
        "B": B,
        "A_sf_sm100": A_sf_sm100,
        "B_sf_sm100": B_sf_sm100,
        "A_sf_sm103": A_sf_sm103,
        "B_sf_sm103": B_sf_sm103,
        "alpha": alpha,
        "D": D,
    }


def create_quant_tensors(
    m: int, n: int, dtype: torch.dtype = torch.bfloat16
) -> dict:
    """Create inputs for activation quantization benchmark."""
    input_tensor = torch.randn(m, n, dtype=dtype, device="cuda")
    global_scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    return {"input": input_tensor, "global_scale": global_scale}


def bench_fn(
    fn,
    warmup: int = 20,
    iters: int = 100,
    sync: bool = True,
) -> float:
    """Benchmark a function, returning median time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    # Timed iterations using CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
    times.sort()
    # Return median in microseconds
    return times[len(times) // 2]


# ============================================================================
# GEMM Benchmark
# ============================================================================


def benchmark_gemm(
    m_sizes: list[int],
    n: int = 7168,
    k: int = 7168,
    dtype: torch.dtype = torch.bfloat16,
) -> list[dict]:
    """
    Benchmark SM100 vs SM103 vs SM103+PDL NVFP4 GEMM kernels side by side.

    PDL on the GEMM sets ProgrammaticStreamSerialization, allowing the NEXT
    kernel on the stream to overlap with the GEMM's tail.  For isolated GEMM
    calls (no consumer kernel), the PDL overhead should be near-zero.
    """
    vllm_ops = torch.ops._C

    has_sm100a = hasattr(vllm_ops, "cutlass_scaled_fp4_mm_sm100a")
    has_sm103a = hasattr(vllm_ops, "cutlass_scaled_fp4_mm_sm103a")
    has_sm103a_pdl = hasattr(vllm_ops, "cutlass_scaled_fp4_mm_sm103a_pdl")

    if not has_sm100a and not has_sm103a:
        print("WARNING: Neither sm100a nor sm103a ops are available. "
              "Rebuild with ENABLE_NVFP4_SM100=1.")
        return []

    results = []

    for m in m_sizes:
        tensors = create_nvfp4_tensors(m, n, k, dtype)
        D = tensors["D"]
        A, B = tensors["A"], tensors["B"]
        A_sf_sm100, B_sf_sm100 = tensors["A_sf_sm100"], tensors["B_sf_sm100"]
        A_sf_sm103, B_sf_sm103 = tensors["A_sf_sm103"], tensors["B_sf_sm103"]
        alpha = tensors["alpha"]

        flops = 2.0 * m * n * k

        time_sm100: Optional[float] = None
        time_sm103: Optional[float] = None
        time_sm103_pdl: Optional[float] = None

        if has_sm100a:
            def run_sm100():
                vllm_ops.cutlass_scaled_fp4_mm_sm100a(
                    D, A, B, A_sf_sm100, B_sf_sm100, alpha
                )
            time_sm100 = bench_fn(run_sm100, warmup=20, iters=100)

        if has_sm103a:
            def run_sm103():
                vllm_ops.cutlass_scaled_fp4_mm_sm103a(
                    D, A, B, A_sf_sm103, B_sf_sm103, alpha
                )
            time_sm103 = bench_fn(run_sm103, warmup=20, iters=100)

        if has_sm103a_pdl:
            def run_sm103_pdl():
                vllm_ops.cutlass_scaled_fp4_mm_sm103a_pdl(
                    D, A, B, A_sf_sm103, B_sf_sm103, alpha
                )
            time_sm103_pdl = bench_fn(run_sm103_pdl, warmup=20, iters=100)

        row: dict = {"M": m, "N": n, "K": k}

        if time_sm100 is not None:
            row["sm100_us"] = time_sm100
            row["sm100_tflops"] = flops / (time_sm100 * 1e-6) / 1e12

        if time_sm103 is not None:
            row["sm103_us"] = time_sm103
            row["sm103_tflops"] = flops / (time_sm103 * 1e-6) / 1e12

        if time_sm103_pdl is not None:
            row["sm103pdl_us"] = time_sm103_pdl
            row["sm103pdl_tflops"] = flops / (time_sm103_pdl * 1e-6) / 1e12

        if time_sm100 is not None and time_sm103 is not None:
            row["sm103_vs_100"] = time_sm100 / time_sm103

        results.append(row)

    return results


# ============================================================================
# Quantization Benchmark
# ============================================================================


def benchmark_quant(
    m_sizes: list[int],
    n: int = 7168,
    dtype: torch.dtype = torch.bfloat16,
) -> list[dict]:
    """
    Benchmark SM100 vs SM103 activation quantization (BF16 -> NVFP4).
    """
    vllm_ops = torch.ops._C
    results = []

    has_sm103_quant = hasattr(vllm_ops, "scaled_fp4_quant_sm103")

    for m in m_sizes:
        tensors = create_quant_tensors(m, n, dtype)
        input_t = tensors["input"]
        global_scale = tensors["global_scale"]

        # SM100 quantization (swizzled layout)
        def run_sm100_quant():
            vllm_ops.scaled_fp4_quant(input_t, global_scale, True)

        time_sm100 = bench_fn(run_sm100_quant, warmup=20, iters=100)

        row: dict = {
            "M": m,
            "N": n,
            "sm100_us": time_sm100,
            "sm100_gb_s": (m * n * 2) / (time_sm100 * 1e-6) / 1e9,
        }

        if has_sm103_quant:
            def run_sm103_quant():
                vllm_ops.scaled_fp4_quant_sm103(input_t, global_scale)

            time_sm103 = bench_fn(run_sm103_quant, warmup=20, iters=100)
            row["sm103_us"] = time_sm103
            row["sm103_gb_s"] = (m * n * 2) / (time_sm103 * 1e-6) / 1e9

        results.append(row)

    return results


# ============================================================================
# SF Layout Conversion Benchmark
# ============================================================================


def benchmark_sf_conversion(
    m_sizes: list[int],
    k: int = 7168,
) -> list[dict]:
    """
    Benchmark the SM100 <-> SM103 scale factor layout conversion kernel.

    This measures the overhead of converting scale factors between layouts,
    which happens once at model load time for weights.
    """
    vllm_ops = torch.ops._C
    results = []

    for m in m_sizes:
        sf_m = round_up(m, 128)
        sf_k = round_up(k // 16, 4)

        # Create source SF tensor (SM100 layout)
        src = torch.randint(
            0, 256, (sf_m, sf_k), dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn)

        # Allocate destination (same shape)
        dst = torch.empty_like(src)

        # Benchmark SM100 -> SM103 conversion
        def run_convert():
            vllm_ops.convert_sf_layout_sm100_to_sm103(dst, src)

        time_us = bench_fn(run_convert, warmup=20, iters=200)

        results.append({
            "M": m,
            "K": k,
            "sf_shape": f"{sf_m}x{sf_k}",
            "kernel": "SM100->SM103 SF convert",
            "time_us": time_us,
            "throughput_gb_s": (sf_m * sf_k) / (time_us * 1e-6) / 1e9,
        })

    return results


# ============================================================================
# End-to-End Benchmark (Quant + GEMM) with PDL comparison
# ============================================================================


def benchmark_e2e(
    m_sizes: list[int],
    n: int = 7168,
    k: int = 7168,
    dtype: torch.dtype = torch.bfloat16,
) -> list[dict]:
    """
    Benchmark the full NVFP4 inference path: quantize activations + GEMM,
    comparing SM100, SM103, and SM103+PDL.

    This measures what a real transformer linear layer does:
    1. Quantize BF16 activations to NVFP4 (with block scales)
    2. NVFP4 x NVFP4 GEMM

    SM103+PDL enables ProgrammaticStreamSerialization on the quant kernel
    and launch_with_pdl on the GEMM, allowing the GEMM to begin executing
    while the quant kernel is still completing its last thread blocks.
    """
    vllm_ops = torch.ops._C

    has_sm100a = hasattr(vllm_ops, "cutlass_scaled_fp4_mm_sm100a")
    has_sm103a = hasattr(vllm_ops, "cutlass_scaled_fp4_mm_sm103a")
    has_sm103_quant = hasattr(vllm_ops, "scaled_fp4_quant_sm103")
    has_sm103_pdl_quant = hasattr(vllm_ops, "scaled_fp4_quant_sm103_pdl")
    has_sm103a_pdl = hasattr(vllm_ops, "cutlass_scaled_fp4_mm_sm103a_pdl")

    if not has_sm100a and not has_sm103a:
        print("WARNING: Neither sm100a nor sm103a ops are available. "
              "Rebuild with ENABLE_NVFP4_SM100=1.")
        return []

    results = []

    for m in m_sizes:
        # Create activation input
        activation = torch.randn(m, k, dtype=dtype, device="cuda")
        global_scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        # Create weight (pre-quantized)
        B = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        sf_n = round_up(n, 128)
        sf_k = round_up(k // 16, 4)

        # Weight SFs in SM100 layout (for SM100 kernel)
        B_sf_sm100 = torch.randint(
            0, 256, (sf_n, sf_k), dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn)
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        # Weight SFs in SM103 layout (pre-converted at load time)
        B_sf_sm103 = torch.empty_like(B_sf_sm100)
        vllm_ops.convert_sf_layout_sm100_to_sm103(B_sf_sm103, B_sf_sm100)

        D = torch.empty(m, n, dtype=dtype, device="cuda")

        flops = 2.0 * m * n * k
        row: dict = {"M": m, "N": n, "K": k}

        # --- SM100 baseline: SM100 quant + SM100 GEMM ---
        if has_sm100a:
            def run_e2e_sm100():
                A_q, A_sf = vllm_ops.scaled_fp4_quant(
                    activation, global_scale, True
                )
                A_sf = A_sf.view(torch.float8_e4m3fn)
                vllm_ops.cutlass_scaled_fp4_mm_sm100a(
                    D, A_q, B, A_sf, B_sf_sm100, alpha
                )

            time_sm100 = bench_fn(run_e2e_sm100, warmup=10, iters=50)
            row["sm100_us"] = time_sm100
            row["sm100_tflops"] = flops / (time_sm100 * 1e-6) / 1e12

        # --- SM103 without PDL: SM103 quant + SM103 GEMM ---
        if has_sm103a and has_sm103_quant:
            def run_e2e_sm103():
                A_q, A_sf = vllm_ops.scaled_fp4_quant_sm103(
                    activation, global_scale
                )
                A_sf = A_sf.view(torch.float8_e4m3fn)
                vllm_ops.cutlass_scaled_fp4_mm_sm103a(
                    D, A_q, B, A_sf, B_sf_sm103, alpha
                )

            time_sm103 = bench_fn(run_e2e_sm103, warmup=10, iters=50)
            row["sm103_us"] = time_sm103
            row["sm103_tflops"] = flops / (time_sm103 * 1e-6) / 1e12

        # --- SM103 with PDL: PDL quant + PDL GEMM ---
        if has_sm103a_pdl and has_sm103_pdl_quant:
            def run_e2e_sm103_pdl():
                # PDL quant: ProgrammaticStreamSerialization allows GEMM to
                # begin before quant finishes.
                A_q, A_sf = vllm_ops.scaled_fp4_quant_sm103_pdl(
                    activation, global_scale
                )
                A_sf = A_sf.view(torch.float8_e4m3fn)
                # PDL GEMM: ProgrammaticStreamSerialization allows the next
                # layer's kernel to begin before this GEMM finishes.
                vllm_ops.cutlass_scaled_fp4_mm_sm103a_pdl(
                    D, A_q, B, A_sf, B_sf_sm103, alpha
                )

            time_sm103_pdl = bench_fn(run_e2e_sm103_pdl, warmup=10, iters=50)
            row["sm103pdl_us"] = time_sm103_pdl
            row["sm103pdl_tflops"] = flops / (time_sm103_pdl * 1e-6) / 1e12

        # Speedup columns
        if "sm100_us" in row and "sm103_us" in row:
            row["sm103_vs_100"] = row["sm100_us"] / row["sm103_us"]
        if "sm103_us" in row and "sm103pdl_us" in row:
            row["pdl_vs_nop"] = row["sm103_us"] / row["sm103pdl_us"]
        if "sm100_us" in row and "sm103pdl_us" in row:
            row["pdl_vs_100"] = row["sm100_us"] / row["sm103pdl_us"]

        results.append(row)

    return results


# ============================================================================
# PDL Pipeline Benchmark (back-to-back quant+GEMM pairs)
# ============================================================================


def benchmark_pdl_pipeline(
    m_sizes: list[int],
    n: int = 7168,
    k: int = 7168,
    num_layers: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> list[dict]:
    """
    Benchmark the PDL pipeline benefit for back-to-back layers.

    In a real transformer, the same quant->GEMM pattern repeats for each
    linear layer.  With PDL enabled on both quant and GEMM, each kernel
    launch overlaps with its predecessor's tail, creating a pipeline:

        quant_1 -> GEMM_1 -> quant_2 -> GEMM_2 -> ...

    This benchmark simulates `num_layers` consecutive quant+GEMM pairs
    to measure the cumulative pipeline benefit.
    """
    vllm_ops = torch.ops._C

    has_sm103_quant = hasattr(vllm_ops, "scaled_fp4_quant_sm103")
    has_sm103_pdl_quant = hasattr(vllm_ops, "scaled_fp4_quant_sm103_pdl")
    has_sm103a = hasattr(vllm_ops, "cutlass_scaled_fp4_mm_sm103a")
    has_sm103a_pdl = hasattr(vllm_ops, "cutlass_scaled_fp4_mm_sm103a_pdl")

    if not (has_sm103_quant and has_sm103a):
        print("WARNING: SM103 ops not available.")
        return []

    results = []

    for m in m_sizes:
        activation = torch.randn(m, k, dtype=dtype, device="cuda")
        global_scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        B = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        sf_n = round_up(n, 128)
        sf_k = round_up(k // 16, 4)
        B_sf_sm100 = torch.randint(
            0, 256, (sf_n, sf_k), dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn)
        B_sf_sm103 = torch.empty_like(B_sf_sm100)
        vllm_ops.convert_sf_layout_sm100_to_sm103(B_sf_sm103, B_sf_sm100)
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        D = torch.empty(m, n, dtype=dtype, device="cuda")

        total_flops = 2.0 * m * n * k * num_layers

        # SM103 without PDL: num_layers sequential quant+GEMM
        def run_pipeline_no_pdl():
            for _ in range(num_layers):
                A_q, A_sf = vllm_ops.scaled_fp4_quant_sm103(
                    activation, global_scale
                )
                A_sf = A_sf.view(torch.float8_e4m3fn)
                vllm_ops.cutlass_scaled_fp4_mm_sm103a(
                    D, A_q, B, A_sf, B_sf_sm103, alpha
                )

        time_no_pdl = bench_fn(run_pipeline_no_pdl, warmup=5, iters=30)

        row: dict = {
            "M": m, "layers": num_layers,
            "no_pdl_us": time_no_pdl,
            "no_pdl_tflops": total_flops / (time_no_pdl * 1e-6) / 1e12,
        }

        # SM103 with PDL: num_layers pipelined quant+GEMM
        if has_sm103_pdl_quant and has_sm103a_pdl:
            def run_pipeline_pdl():
                for _ in range(num_layers):
                    A_q, A_sf = vllm_ops.scaled_fp4_quant_sm103_pdl(
                        activation, global_scale
                    )
                    A_sf = A_sf.view(torch.float8_e4m3fn)
                    vllm_ops.cutlass_scaled_fp4_mm_sm103a_pdl(
                        D, A_q, B, A_sf, B_sf_sm103, alpha
                    )

            time_pdl = bench_fn(run_pipeline_pdl, warmup=5, iters=30)
            row["pdl_us"] = time_pdl
            row["pdl_tflops"] = total_flops / (time_pdl * 1e-6) / 1e12
            row["pdl_speedup"] = time_no_pdl / time_pdl

        results.append(row)

    return results


# ============================================================================
# Main
# ============================================================================


def print_results(results: list[dict], title: str):
    if not results:
        return

    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

    # Determine columns from first result
    cols = list(results[0].keys())
    # Header
    header = " | ".join(f"{c:>15s}" for c in cols)
    print(header)
    print("-" * len(header))

    for r in results:
        row = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                row.append(f"{v:>15.2f}")
            elif isinstance(v, int):
                row.append(f"{v:>15d}")
            else:
                row.append(f"{v:>15s}")
        print(" | ".join(row))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 SM103 vs SM100 kernels (with PDL)"
    )
    parser.add_argument(
        "--mode",
        choices=["gemm", "quant", "sf_convert", "e2e", "pdl", "all"],
        default="all",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--n", type=int, default=7168,
        help="N dimension (default: 7168, DeepSeek)",
    )
    parser.add_argument(
        "--k", type=int, default=7168,
        help="K dimension (default: 7168, DeepSeek)",
    )
    parser.add_argument(
        "--layers", type=int, default=4,
        help="Number of back-to-back layers for PDL pipeline benchmark",
    )
    args = parser.parse_args()

    sm = get_sm_version()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM version: {sm}")
    print(f"CUDA version: {torch.version.cuda}")

    if sm < 100:
        print("ERROR: This benchmark requires SM100+ (Blackwell) GPU.")
        return

    if sm == 103:
        print("NOTE: Running on SM103 (B300) -- all kernel variants will run.")
    else:
        print(f"NOTE: Running on SM{sm} -- SM100 kernel is native; "
              "SM103 kernel runs via forward compat (may be slower).")

    # Problem sizes typical for LLM inference
    # Small M = decode, large M = prefill
    m_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    if args.mode in ("gemm", "all"):
        results = benchmark_gemm(m_sizes, n=args.n, k=args.k)
        print_results(
            results,
            f"NVFP4 GEMM: SM100 vs SM103 vs SM103+PDL (N={args.n}, K={args.k})",
        )

    if args.mode in ("quant", "all"):
        results = benchmark_quant(m_sizes, n=args.k)
        print_results(results, f"NVFP4 Activation Quantization (N={args.k})")

    if args.mode in ("sf_convert", "all"):
        sf_m_sizes = [1024, 2048, 4096, 7168, 8192, 14336, 16384]
        results = benchmark_sf_conversion(sf_m_sizes, k=args.k)
        print_results(results, "SF Layout Conversion SM100 <-> SM103")

    if args.mode in ("e2e", "all"):
        results = benchmark_e2e(m_sizes, n=args.n, k=args.k)
        print_results(
            results,
            f"E2E NVFP4 (Quant+GEMM): SM100 vs SM103 vs SM103+PDL "
            f"(N={args.n}, K={args.k})",
        )
        print(
            "\nNOTE: sm103_vs_100 = SM100_time / SM103_time (>1 means SM103 faster)\n"
            "      pdl_vs_nop  = SM103_time / SM103+PDL_time (>1 means PDL faster)\n"
            "      pdl_vs_100  = SM100_time / SM103+PDL_time (total speedup)"
        )

    if args.mode in ("pdl", "all"):
        results = benchmark_pdl_pipeline(
            m_sizes, n=args.n, k=args.k, num_layers=args.layers
        )
        print_results(
            results,
            f"PDL Pipeline ({args.layers} layers): SM103 vs SM103+PDL "
            f"(N={args.n}, K={args.k})",
        )
        print(
            "\nNOTE: pdl_speedup = no_pdl_time / pdl_time\n"
            "      PDL overlaps quant tail with GEMM head across layer boundaries.\n"
            "      Benefit is most visible with multiple back-to-back layers."
        )


if __name__ == "__main__":
    main()
