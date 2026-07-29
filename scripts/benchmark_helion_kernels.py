#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark a registered Helion kernel against a baseline.

For each input case produced by the kernel's registered input generator, this
measures the latency of the Helion kernel and a chosen baseline, then reports
the speedup.

Two baselines are supported (``--baseline``):

- ``autotune`` (default): the kernel's autotuning baseline
  (``helion_settings.autotune_baseline_fn``), wrapped in ``torch.compile``
  (inductor). This is the native-torch reference used by kernel autotuning
  and correctness unit tests.
- ``cuda``: the corresponding hand-written CUDA op (``torch.ops._C.*``). The
  mapping from Helion kernel name to CUDA op lives in ``CUDA_BASELINE_OPS``
  below. Every Helion kernel shares the same argument interface as its CUDA
  counterpart, so inputs are forwarded verbatim.

Usage:
    # List available kernels
    python scripts/benchmark_helion_kernels.py --list

    # Benchmark a kernel against the autotune baseline (default)
    python scripts/benchmark_helion_kernels.py --kernel per_token_group_fp8_quant

    # Benchmark against the CUDA baseline
    python scripts/benchmark_helion_kernels.py --kernel per_token_group_fp8_quant \\
        --baseline cuda

    # Disable CUDA graph capture and save results
    python scripts/benchmark_helion_kernels.py --kernel per_token_group_fp8_quant \\
        --no-cudagraph --output results.json
"""

import argparse
import copy
import gc
import json
import statistics
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from vllm.triton_utils import triton

try:
    from vllm.benchmarks.lib.utils import default_vllm_config
    from vllm.kernels.helion import get_kernel_by_name, get_registered_kernels
    from vllm.kernels.helion.ops import import_all_kernels
    from vllm.logger import init_logger
    from vllm.utils.import_utils import has_helion
except ImportError as e:
    print(f"Error importing vLLM: {e}")
    print("Please ensure vLLM is installed and in your Python path")
    sys.exit(1)

logger = init_logger("vllm.scripts.benchmark_helion_kernels")


# Maps a Helion kernel name to the CUDA op (attribute on ``torch.ops._C``) that
# implements the same operation. Helion kernels share the CUDA op's argument
# interface, so the kernel's input tuple is forwarded verbatim. Add an entry
# here when introducing a new kernel whose baseline should be the CUDA op.
CUDA_BASELINE_OPS: dict[str, str] = {
    "dynamic_per_token_scaled_fp8_quant": "dynamic_per_token_scaled_fp8_quant",
    "fused_qk_norm_rope": "fused_qk_norm_rope",
    "per_token_group_fp8_quant": "per_token_group_fp8_quant",
    "rms_norm_dynamic_per_token_quant": "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_block_quant": "rms_norm_per_block_quant",
    "silu_and_mul_per_block_quant": "silu_and_mul_per_block_quant",
    "scaled_mm": "cutlass_scaled_mm",
}

# torch.compile options for the torch baseline, mirroring how these kernels are
# compiled inside vLLM.
_TORCH_COMPILE_OPTIONS: dict[str, bool] = {
    "enable_auto_functionalized_v2": False,
    "size_asserts": False,
    "alignment_asserts": False,
    "scalar_asserts": False,
    "combo_kernels": True,
    "benchmark_combo_kernel": True,
}


@dataclass
class Row:
    case: str
    baseline_ms: float
    kernel_ms: float
    speedup_x: float


def print_table(rows: list[Row]) -> None:
    headers = ["case", "baseline_ms", "kernel_ms", "speedup(x)"]

    data = [
        [
            r.case,
            f"{r.baseline_ms:.3f}",
            f"{r.kernel_ms:.3f}",
            f"{r.speedup_x:.3f}",
        ]
        for r in rows
    ]

    cols = list(zip(*([headers] + data)))
    widths = [max(len(cell) for cell in col) for col in cols]

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))


def list_kernels() -> None:
    kernels = get_registered_kernels()

    if not kernels:
        print("No Helion kernels found in registry.")
        return

    print("Available Helion kernels:")
    print("=" * 50)
    for name in sorted(kernels.keys()):
        cuda = CUDA_BASELINE_OPS.get(name)
        suffix = "" if cuda else "  (no CUDA baseline mapping)"
        print(f"  {name}{suffix}")
    print(f"\nTotal: {len(kernels)} kernels")


def check_requirements() -> bool:
    if not torch.accelerator.is_available():
        logger.error("CUDA is not available. Helion benchmarking requires GPU.")
        return False
    if not has_helion():
        logger.error("Helion is not installed. Please install Helion package.")
        return False
    return True


def make_cuda_baseline(kernel_name: str) -> Callable:
    """Return a callable invoking the CUDA op mapped to ``kernel_name``.

    The Helion kernel and its CUDA op share the same argument interface, so the
    input tuple is forwarded verbatim.
    """
    cuda_op_name = CUDA_BASELINE_OPS.get(kernel_name)
    if cuda_op_name is None:
        logger.error(
            "No CUDA baseline mapping for kernel '%s'. Add an entry to "
            "CUDA_BASELINE_OPS in %s (mapping the kernel name to its "
            "torch.ops._C.<op> name), or benchmark with --baseline torch.",
            kernel_name,
            __file__,
        )
        sys.exit(1)

    cuda_op = getattr(torch.ops._C, cuda_op_name, None)
    if cuda_op is None:
        logger.error(
            "torch.ops._C.%s is not available. Ensure the vLLM C extension is "
            "built and loaded.",
            cuda_op_name,
        )
        sys.exit(1)

    return cuda_op


def make_autotune_baseline(kernel_name: str) -> Callable:
    """Return the kernel's autotune baseline wrapped in ``torch.compile``.

    The baseline is the native-torch reference the kernel is tuned against,
    registered via ``helion_settings.autotune_baseline_fn``.
    """
    wrapper = get_kernel_by_name(kernel_name)
    settings = wrapper.helion_settings
    baseline_fn = getattr(settings, "autotune_baseline_fn", None)
    if baseline_fn is None:
        logger.error(
            "Kernel '%s' has no autotune_baseline_fn in its helion_settings, so "
            "the 'autotune' baseline is unavailable. Register one via "
            "register_kernel(..., helion_settings=helion.Settings("
            "autotune_baseline_fn=...)), or benchmark with --baseline cuda.",
            kernel_name,
        )
        sys.exit(1)

    return torch.compile(
        baseline_fn,
        fullgraph=True,
        dynamic=False,
        backend="inductor",
        options=_TORCH_COMPILE_OPTIONS,
    )


def cleanup_gpu_resources() -> None:
    try:
        torch.accelerator.empty_cache()
        gc.collect()
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        torch.accelerator.synchronize()
    except Exception as e:
        logger.warning("Failed to cleanup GPU resources: %s", e)


_REDUCERS: dict[str, Callable[[list[float]], float]] = {
    "min": min,
    "max": max,
    "mean": statistics.fmean,
    "median": statistics.median,
}


def _reduce(times: list[float], return_mode: str) -> float:
    return _REDUCERS[return_mode](times)


def do_bench_cudagraph_l2_clear(
    fn: Callable, rep: int = 100, return_mode: str = "mean"
) -> float:
    """CUDA-graph benchmark that flushes the L2 cache before every call.

    ``triton.testing.do_bench_cudagraph`` captures back-to-back kernel launches
    with a warm L2 cache, which over-estimates performance for memory-bound
    kernels. This clears L2 (via triton's benchmark cache buffer) before each
    call and subtracts the isolated cache-clear cost from the measured time.

    Adapted from tritonbench's ``_do_bench_cudagraph_with_cache_clear`` using
    only triton/torch primitives so no extra dependency is introduced.
    """
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    clear_cache = cache.zero_

    s = torch.Stream()
    with s:
        clear_cache()
        fn()

        start_event = torch.Event(enable_timing=True)
        end_event = torch.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            clear_cache()
            fn()
        end_event.record()
        torch.accelerator.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        n_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(n_repeat):
                clear_cache()
                fn()

        clear_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(clear_graph):
            for _ in range(n_repeat):
                clear_cache()
        torch.accelerator.synchronize()

        kernel_times = []
        for _ in range(10):
            start_event = torch.Event(enable_timing=True)
            end_event = torch.Event(enable_timing=True)
            start_event.record()
            clear_graph.replay()
            end_event.record()
            torch.accelerator.synchronize()
            clear_ms = start_event.elapsed_time(end_event) / n_repeat

            start_event = torch.Event(enable_timing=True)
            end_event = torch.Event(enable_timing=True)
            start_event.record()
            graph.replay()
            end_event.record()
            torch.accelerator.synchronize()
            total_ms = start_event.elapsed_time(end_event) / n_repeat

            kernel_times.append(total_ms - clear_ms)

    return _reduce(kernel_times, return_mode)


@torch.inference_mode()
def benchmark(
    kernel_name: str,
    baseline_fn: Callable,
    repeat: int,
    cudagraph: bool,
    return_mode: str,
) -> list[Row]:
    kernel = get_kernel_by_name(kernel_name)
    # do_bench already flushes L2 per call; do_bench_cudagraph does not, so use
    # the cache-clearing variant to avoid warm-L2 over-estimates.
    benchmark_fn = do_bench_cudagraph_l2_clear if cudagraph else triton.testing.do_bench

    inputs_dict = kernel.get_inputs()
    rows: list[Row] = []

    for key, inputs in inputs_dict.items():
        logger.info("Benchmarking case %s", key)

        # Kernels may mutate their inputs in place; give each side its own copy.
        kernel_inputs = copy.deepcopy(inputs)
        baseline_inputs = copy.deepcopy(inputs)

        kernel_latency = benchmark_fn(
            lambda kernel_inputs=kernel_inputs: kernel(*kernel_inputs),
            rep=repeat,
            return_mode=return_mode,
        )
        baseline_latency = benchmark_fn(
            lambda baseline_inputs=baseline_inputs: baseline_fn(*baseline_inputs),
            rep=repeat,
            return_mode=return_mode,
        )

        rows.append(
            Row(
                case=str(key),
                baseline_ms=baseline_latency,
                kernel_ms=kernel_latency,
                speedup_x=baseline_latency / kernel_latency,
            )
        )
        cleanup_gpu_resources()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a Helion kernel against a baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available Helion kernels and exit",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        help="Name of the single Helion kernel to benchmark",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Number of benchmark repetitions (default: 100)",
    )
    parser.add_argument(
        "--no-cudagraph",
        dest="cudagraph",
        action="store_false",
        help="Disable CUDA graph mode (enabled by default)",
    )
    parser.add_argument(
        "--baseline",
        choices=["cuda", "autotune"],
        default="autotune",
        help=(
            "Baseline to compare against: 'autotune' uses the kernel's "
            "autotune_baseline_fn under torch.compile; 'cuda' uses the mapped "
            "torch.ops._C op (default: autotune)"
        ),
    )
    parser.add_argument(
        "--return-mode",
        choices=["min", "max", "mean", "median"],
        default="mean",
        help="Statistic to report from the benchmark samples (default: mean)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save benchmark results as JSON (default: log only)",
    )

    args = parser.parse_args()

    import_all_kernels()

    if args.list:
        list_kernels()
        return

    if not args.kernel:
        parser.error("--kernel is required (or use --list to see available kernels)")

    kernels = get_registered_kernels()
    if args.kernel not in kernels:
        logger.error("Kernel '%s' not found in registry.", args.kernel)
        logger.error("Available kernels: %s", sorted(kernels.keys()))
        sys.exit(1)

    wrapper = kernels[args.kernel]
    if wrapper._disabled:
        logger.error(
            "Kernel '%s' is disabled: %s",
            args.kernel,
            wrapper._disabled_reason,
        )
        sys.exit(1)

    if not check_requirements():
        sys.exit(1)

    with default_vllm_config():
        if args.baseline == "cuda":
            baseline_fn = make_cuda_baseline(args.kernel)
        else:
            baseline_fn = make_autotune_baseline(args.kernel)

        rows = benchmark(
            args.kernel,
            baseline_fn,
            args.repeat,
            args.cudagraph,
            args.return_mode,
        )

    print_table(rows)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "kernel": args.kernel,
                    "baseline": args.baseline,
                    "cudagraph": args.cudagraph,
                    "repeat": args.repeat,
                    "return_mode": args.return_mode,
                    "results": [asdict(r) for r in rows],
                },
                f,
                indent=2,
            )
        logger.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
