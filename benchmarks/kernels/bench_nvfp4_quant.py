# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import copy
import itertools

import torch
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    convert_swizzled_to_linear,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton
from vllm.utils.flashinfer import flashinfer_fp4_quantize

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 requires compute capability of 10.0 (Blackwell)")

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

PROVIDER_CFGS = {
    "vllm": dict(backend="vllm", enabled=True),
    "flashinfer": dict(backend="flashinfer", enabled=True),
}

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]


def compute_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Compute global scale for FP4 quantization."""
    amax = torch.abs(tensor).max().to(torch.float32)
    return FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled,
        line_names=_enabled,
        ylabel="us (lower is better)",
        plot_name="NVFP4 Input Quantization Latency (us)",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    M = batch_size
    device = "cuda"
    dtype = torch.bfloat16

    # Create input tensor
    a = torch.randn((M, K), device=device, dtype=dtype)

    # Compute global scale for activation
    a_global_scale = compute_global_scale(a)

    quantiles = [0.5, 0.2, 0.8]

    cfg = PROVIDER_CFGS[provider]

    if cfg["backend"] == "vllm":
        # vLLM's FP4 quantization
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: ops.scaled_fp4_quant(a, a_global_scale),
            quantiles=quantiles,
        )
    elif cfg["backend"] == "flashinfer":
        # FlashInfer's FP4 quantization
        # Use is_sf_swizzled_layout=True to match vLLM's output format
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: flashinfer_fp4_quantize(
                a, a_global_scale, is_sf_swizzled_layout=True
            ),
            quantiles=quantiles,
        )

    # Convert ms to us for better readability at small batch sizes
    to_us = lambda t_ms: t_ms * 1000
    return to_us(ms), to_us(max_ms), to_us(min_ms)


def prepare_shapes(args):
    out = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            KN.append(model)
            out.append(KN)
    return out


def _test_accuracy_once(M: int, K: int, dtype: torch.dtype, device: str):
    """Test accuracy between vLLM and FlashInfer FP4 quantization."""
    # Create input tensor
    a = torch.randn((M, K), device=device, dtype=dtype)

    # Compute global scale
    a_global_scale = compute_global_scale(a)

    # vLLM quantization
    vllm_fp4, vllm_scale = ops.scaled_fp4_quant(a, a_global_scale)

    # FlashInfer quantization (with swizzled layout to match vLLM's output)
    flashinfer_fp4, flashinfer_scale = flashinfer_fp4_quantize(
        a, a_global_scale, is_sf_swizzled_layout=True
    )
    flashinfer_scale = flashinfer_scale.view(torch.float8_e4m3fn)

    # Compare outputs
    torch.testing.assert_close(
        vllm_fp4,
        flashinfer_fp4,
    )
    print(f"M={M}, K={K}, dtype={dtype}: PASSED")


def test_accuracy():
    """Run accuracy tests across various shapes."""
    print("\n" + "=" * 60)
    print("Running accuracy tests: vLLM vs FlashInfer")
    print("=" * 60)

    device = "cuda"
    dtype = torch.bfloat16

    # Test various batch sizes and hidden dimensions
    Ms = [1, 1024]
    Ks = [4096]

    for M in Ms:
        for K in Ks:
            _test_accuracy_once(M, K, dtype, device)

    print("\nAll accuracy tests passed!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="method",
        line_vals=[
            "flashinfer_swizzled_no_convert",
            "flashinfer_swizzled_convert",
            "vllm_swizzled_convert",
        ],
        line_names=[
            "FlashInfer (swizzled, no convert)",
            "FlashInfer (swizzled + convert)",
            "vLLM (swizzled + convert)",
        ],
        ylabel="us (lower is better)",
        plot_name="NVFP4 E2E Quantization + Conversion Latency (us)",
        args={},
    )
)
def benchmark_e2e_quantization(batch_size, method, N, K):
    """
    End-to-end benchmark: quantization with/without conversion.

    This tests different paths:
    - FlashInfer swizzled (no convert): for kernels needing swizzled layout
    - FlashInfer swizzled + convert: for kernels needing linear layout (TRTLLM)
    - vLLM swizzled + convert: current path for TRTLLM kernels

    Following reviewer's suggestion: force FlashInfer to swizzle the SF layout
    rather than linearizing the vLLM quantized output, as swizzling is actually
    needed e2e (for some kernels like FlashInfer CUTLASS).
    """
    M = batch_size
    device = "cuda"
    dtype = torch.bfloat16

    # Create input tensor
    a = torch.randn((M, K), device=device, dtype=dtype)

    # Compute global scale for activation
    a_global_scale = compute_global_scale(a)

    quantiles = [0.5, 0.2, 0.8]

    if method == "flashinfer_swizzled_no_convert":
        # FlashInfer quantization with swizzled layout, no conversion
        # (for kernels that need swizzled layout, e.g., FlashInfer CUTLASS)
        def fn():
            fp4, scale_swizzled = flashinfer_fp4_quantize(
                a, a_global_scale, is_sf_swizzled_layout=True
            )
            scale_swizzled = scale_swizzled.view(torch.float8_e4m3fn)
            # No conversion - use swizzled layout directly (needed e2e)
            return fp4, scale_swizzled

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    elif method == "flashinfer_swizzled_convert":
        # FlashInfer quantization with swizzled layout, then convert to linear
        # (for kernels that need linear layout, e.g., TRTLLM)
        def fn():
            fp4, scale_swizzled = flashinfer_fp4_quantize(
                a, a_global_scale, is_sf_swizzled_layout=True
            )
            scale_swizzled = scale_swizzled.view(torch.float8_e4m3fn)
            # Convert swizzled to linear (required by TRTLLM kernels)
            scale_linear = convert_swizzled_to_linear(
                scale_swizzled, M, K, block_size=16
            )
            return fp4, scale_linear

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    elif method == "vllm_swizzled_convert":
        # vLLM quantization (outputs swizzled), then convert to linear
        def fn():
            fp4, scale_swizzled = ops.scaled_fp4_quant(a, a_global_scale)
            # Convert swizzled to linear (required by TRTLLM kernels)
            scale_linear = convert_swizzled_to_linear(
                scale_swizzled, M, K, block_size=16
            )
            return fp4, scale_linear

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    # Convert ms to us for better readability at small batch sizes
    to_us = lambda t_ms: t_ms * 1000
    return to_us(ms), to_us(max_ms), to_us(min_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 quantization: vLLM vs FlashInfer"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.1-8B-Instruct"],
        choices=list(WEIGHT_SHAPES.keys()),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save benchmark results",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run accuracy tests",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run end-to-end benchmark (quantization + conversion)",
    )
    args = parser.parse_args()

    if args.accuracy:
        test_accuracy()

    if args.e2e:
        print("\n" + "=" * 80)
        print("Running E2E Benchmark: Quantization + Conversion to Linear")
        print("=" * 80)
        print("This benchmark tests different quantization paths:")
        print(
            "  - FlashInfer swizzled (no convert): for kernels needing swizzled"
            " layout (e.g., FlashInfer CUTLASS)"
        )
        print(
            "  - FlashInfer swizzled + convert: for kernels needing linear"
            " layout (e.g., TRTLLM)"
        )
        print("  - vLLM swizzled + convert: current path for TRTLLM kernels")
        print(
            "\nFollowing reviewer's suggestion: force FlashInfer to swizzle"
            " the SF layout rather than linearizing the vLLM quantized output,"
            " as swizzling is actually needed e2e."
        )
        print("=" * 80)

        for K, N, model in prepare_shapes(args):
            print(f"\n{model}, N={N} K={K}")
            benchmark_e2e_quantization.run(
                print_data=True,
                save_path=args.save_path,
                N=N,
                K=K,
            )
    else:
        for K, N, model in prepare_shapes(args):
            print(f"\n{model}, N={N} K={K}")
            benchmark.run(
                print_data=True,
                save_path=args.save_path,
                N=N,
                K=K,
            )

    print("\nBenchmark finished!")
