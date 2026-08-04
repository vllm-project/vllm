# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark NVFP4 input quantization kernels.

Two comparisons are reported:

- Main plot: vLLM's built-in C++ kernel versus FlashInfer's CUDA and CuTe-DSL
  kernels, for the linear and 128x4 swizzled scale layouts. The vLLM and
  FlashInfer CuTe-DSL lines are the production paths reachable through
  scaled_fp4_quant; the FlashInfer CUDA line is an external reference kernel
  (vLLM does not dispatch to it for these layouts) called directly for
  comparison, as this benchmark has done since it was first added.
- 8x4 small-M plot: the TRTLLM 8x4 scale layout, CUDA versus CuTe-DSL. Both are
  production paths; scaled_fp4_quant selects this layout only when gemm_backend
  contains "trtllm" and M <= 32.
"""

import argparse
import copy
import itertools
import os

import torch
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton
from vllm.utils.flashinfer import (
    flashinfer_fp4_quantize,
    has_flashinfer,
    has_flashinfer_cutedsl_nvfp4_quant,
)

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 requires compute capability of 10.0 (Blackwell)")

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

_flashinfer_ok = has_flashinfer()
_flashinfer_cutedsl_ok = has_flashinfer_cutedsl_nvfp4_quant()

# Main comparison: three NVFP4 quant implementations across the linear and 128x4
# swizzled scale layouts. ``backend`` selects the implementation and ``swizzle``
# selects the 128x4 layout. ``vllm`` and ``flashinfer-cutedsl`` are the production
# paths reachable through scaled_fp4_quant; ``flashinfer-cuda`` is FlashInfer's
# standalone CUDA kernel, kept as an external reference (see module docstring).
PROVIDER_CFGS = {
    "vllm-linear": dict(backend="vllm", swizzle=False, enabled=True),
    "vllm-swizzle": dict(backend="vllm", swizzle=True, enabled=True),
    "flashinfer-cuda-linear": dict(
        backend="flashinfer-cuda", swizzle=False, enabled=_flashinfer_ok
    ),
    "flashinfer-cuda-swizzle": dict(
        backend="flashinfer-cuda", swizzle=True, enabled=_flashinfer_ok
    ),
    "flashinfer-cutedsl-linear": dict(
        backend="flashinfer-cutedsl", swizzle=False, enabled=_flashinfer_cutedsl_ok
    ),
    "flashinfer-cutedsl-swizzle": dict(
        backend="flashinfer-cutedsl", swizzle=True, enabled=_flashinfer_cutedsl_ok
    ),
}

# Small-M comparison: the TRTLLM 8x4 scale layout only, CUDA vs CuTe-DSL kernel.
# scaled_fp4_quant selects the 8x4 layout when gemm_backend contains "trtllm" and
# M <= 32, so this runs on a separate small-M axis (see benchmark_8x4).
PROVIDER_CFGS_8X4 = {
    "flashinfer-cuda-8x4": dict(quant_backend="auto", enabled=_flashinfer_ok),
    "flashinfer-cutedsl-8x4": dict(
        quant_backend="flashinfer_cutedsl", enabled=_flashinfer_cutedsl_ok
    ),
}

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]
_enabled_8x4 = [k for k, v in PROVIDER_CFGS_8X4.items() if v["enabled"]]


def compute_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Compute global scale for FP4 quantization."""
    amax = torch.abs(tensor).max().to(torch.float32)
    return FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax


def _bench_quant(fn):
    """Time a quantization callable, returning (median, hi, lo) in us."""
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    # Convert ms to us for better readability at small batch sizes
    to_us = lambda t_ms: t_ms * 1000
    return to_us(ms), to_us(max_ms), to_us(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled,
        line_names=_enabled,
        ylabel="us (lower is better)",
        plot_name="nvfp4-input-quant",
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

    cfg = PROVIDER_CFGS[provider]
    if cfg["backend"] == "vllm":
        fn = lambda: ops.scaled_fp4_quant(
            a, a_global_scale, is_sf_swizzled_layout=cfg["swizzle"]
        )
    elif cfg["backend"] == "flashinfer-cuda":
        # FlashInfer's standalone CUDA kernel, called directly (vLLM does not
        # dispatch to it for the linear and 128x4 layouts); kept as an external
        # reference.
        fn = lambda: flashinfer_fp4_quantize(
            a, a_global_scale, is_sf_swizzled_layout=cfg["swizzle"]
        )
    elif cfg["backend"] == "flashinfer-cutedsl":
        fn = lambda: ops.scaled_fp4_quant(
            a,
            a_global_scale,
            is_sf_swizzled_layout=cfg["swizzle"],
            quant_backend="flashinfer_cutedsl",
        )
    else:
        raise ValueError(f"unknown provider backend: {cfg['backend']}")
    return _bench_quant(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        # The 8x4 scale layout is only selected for M <= 32, so restrict the axis
        # to that regime instead of reporting values production never uses.
        x_vals=[1, 4, 8, 16, 32],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled_8x4,
        line_names=_enabled_8x4,
        ylabel="us (lower is better)",
        plot_name="nvfp4-8x4-small-m-quant",
        args={},
    )
)
def benchmark_8x4(batch_size, provider, N, K):
    M = batch_size
    device = "cuda"
    dtype = torch.bfloat16

    # Create input tensor
    a = torch.randn((M, K), device=device, dtype=dtype)

    # Compute global scale for activation
    a_global_scale = compute_global_scale(a)

    cfg = PROVIDER_CFGS_8X4[provider]
    # gemm_backend="trtllm" with M <= 32 selects the 8x4 scale layout; quant_backend
    # picks the CuTe-DSL kernel over the CUDA one.
    fn = lambda: ops.scaled_fp4_quant(
        a,
        a_global_scale,
        gemm_backend="trtllm",
        quant_backend=cfg["quant_backend"],
    )
    return _bench_quant(fn)


def prepare_shapes(args):
    out = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            KN.append(model)
            out.append(KN)
    return out


def _test_accuracy_once(
    M: int, K: int, dtype: torch.dtype, device: str, is_sf_swizzled_layout: bool
):
    """Test accuracy between vLLM and FlashInfer's CUDA FP4 quantization."""
    if not has_flashinfer():
        print("FlashInfer unavailable; skipping accuracy test.")
        return

    # Create input tensor
    a = torch.randn((M, K), device=device, dtype=dtype)

    # Compute global scale
    a_global_scale = compute_global_scale(a)

    # vLLM quantization
    vllm_fp4, vllm_scale = ops.scaled_fp4_quant(
        a, a_global_scale, is_sf_swizzled_layout=is_sf_swizzled_layout
    )

    # FlashInfer CUDA quantization (swizzled layout to match vLLM's output)
    flashinfer_fp4, flashinfer_scale = flashinfer_fp4_quantize(
        a, a_global_scale, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    flashinfer_scale = flashinfer_scale.view(torch.float8_e4m3fn)

    # vLLM's built-in kernel and FlashInfer's CUDA kernel both use an exact
    # reciprocal, so require bit-exact fp4 codes and scales. CuTe-DSL correctness
    # is covered separately by the kernel tests in tests/kernels/quantization.
    torch.testing.assert_close(vllm_fp4, flashinfer_fp4)
    torch.testing.assert_close(vllm_scale, flashinfer_scale)
    print(
        f"M={M}, K={K}, dtype={dtype}, "
        f"is_sf_swizzled_layout={is_sf_swizzled_layout}: PASSED"
    )


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

    for is_sf_swizzled_layout in [True, False]:
        for M in Ms:
            for K in Ks:
                _test_accuracy_once(M, K, dtype, device, is_sf_swizzled_layout)

    print("\nAll accuracy tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 quantization: vLLM vs FlashInfer"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.3-70B-Instruct"],
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
    args = parser.parse_args()

    if args.accuracy:
        test_accuracy()

    for K, N, model in prepare_shapes(args):
        print(f"\n{model}, N={N} K={K}")
        save_path = args.save_path
        if save_path is not None:
            save_path = os.path.join(save_path, f"n{N}_k{K}")
            os.makedirs(save_path, exist_ok=True)
        benchmark.run(
            print_data=True,
            save_path=save_path,
            N=N,
            K=K,
        )
        if _enabled_8x4:
            print(f"\n{model}, N={N} K={K} (8x4 small-M)")
            benchmark_8x4.run(
                print_data=True,
                save_path=save_path,
                N=N,
                K=K,
            )

    print("\nBenchmark finished!")
