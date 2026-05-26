# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 requires compute capability of 10.0 (Blackwell)")


FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

PROVIDER_CFGS = {
    "torch-bf16": dict(enabled=True),
    "nvfp4": dict(no_a_quant=False, enabled=True),
    "nvfp4-noquant": dict(no_a_quant=True, enabled=True),
    "fbgemm-nvfp4": dict(fbgemm=True, no_a_quant=False, enabled=True),
    "fbgemm-nvfp4-noquant": dict(fbgemm=True, no_a_quant=True, enabled=True),
}

_needs_fbgemm = any(
    v.get("fbgemm", False) for v in PROVIDER_CFGS.values() if v.get("enabled", False)
)
if _needs_fbgemm:
    try:
        from fbgemm_gpu.experimental.gemm.triton_gemm.fp4_quantize import (
            triton_scale_nvfp4_quant,
        )
    except ImportError:
        print(
            "WARNING: FBGEMM providers are enabled but fbgemm_gpu is not installed. "
            "These providers will be skipped. Please install fbgemm_gpu with: "
            "'pip install fbgemm-gpu-genai' to run them."
        )
        # Disable FBGEMM providers so the benchmark can run.
        for cfg in PROVIDER_CFGS.values():
            if cfg.get("fbgemm"):
                cfg["enabled"] = False

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]


def _quant_weight_nvfp4(b: torch.Tensor, device: str, cfg):
    # Compute global scale for weight
    b_amax = torch.abs(b).max().to(torch.float32)
    b_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
    if "fbgemm" in cfg and cfg["fbgemm"]:
        b_fp4, scale_b_fp4 = triton_scale_nvfp4_quant(b, b_global_scale)
    else:
        b_fp4, scale_b_fp4 = ops.scaled_fp4_quant(b, b_global_scale)
    return b_fp4, scale_b_fp4, b_global_scale


def build_nvfp4_runner(cfg, a, b, dtype, device):
    b_fp4, scale_b_fp4, b_global_scale = _quant_weight_nvfp4(b, device, cfg)

    # Compute global scale for activation
    # NOTE: This is generally provided ahead-of-time by the model checkpoint.
    a_amax = torch.abs(a).max().to(torch.float32)
    a_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax

    # Alpha for the GEMM operation
    alpha = 1.0 / (a_global_scale * b_global_scale)
    if "fbgemm" in cfg and cfg["fbgemm"]:
        if cfg["no_a_quant"]:
            a_fp4, scale_a_fp4 = triton_scale_nvfp4_quant(a, a_global_scale)

            def run():
                return torch.ops.fbgemm.f4f4bf16(
                    a_fp4,
                    b_fp4,
                    scale_a_fp4,
                    scale_b_fp4,
                    global_scale=alpha,
                    use_mx=False,
                )

            return run
        else:

            def run():
                a_fp4, scale_a_fp4 = triton_scale_nvfp4_quant(a, a_global_scale)
                return torch.ops.fbgemm.f4f4bf16(
                    a_fp4,
                    b_fp4,
                    scale_a_fp4,
                    scale_b_fp4,
                    global_scale=alpha,
                    use_mx=False,
                )

            return run

    if cfg["no_a_quant"]:
        # Pre-quantize activation
        a_fp4, scale_a_fp4 = ops.scaled_fp4_quant(a, a_global_scale)

        def run():
            return ops.cutlass_scaled_fp4_mm(
                a_fp4, b_fp4, scale_a_fp4, scale_b_fp4, alpha, dtype
            )

        return run

    # Quantize activation on-the-fly
    def run():
        a_fp4, scale_a_fp4 = ops.scaled_fp4_quant(a, a_global_scale)
        return ops.cutlass_scaled_fp4_mm(
            a_fp4, b_fp4, scale_a_fp4, scale_b_fp4, alpha, dtype
        )

    return run


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled,
        line_names=_enabled,
        ylabel="TFLOP/s (larger is better)",
        plot_name="BF16 vs NVFP4 GEMMs",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    M = batch_size
    device = "cuda"
    dtype = torch.bfloat16

    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((N, K), device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-bf16":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, b), quantiles=quantiles
        )
    else:
        cfg = PROVIDER_CFGS[provider]
        run_quant = build_nvfp4_runner(cfg, a, b, dtype, device)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: run_quant(), quantiles=quantiles
        )

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


def prepare_shapes(args):
    out = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            KN.append(model)
            out.append(KN)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.1-8B-Instruct"],
        choices=list(WEIGHT_SHAPES.keys()),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    args = parser.parse_args()

    for K, N, model in prepare_shapes(args):
        print(f"{model}, N={N} K={K}, BF16 vs NVFP4 GEMMs TFLOP/s:")
        save_dir = f"bench_nvfp4_res_n{N}_k{K}"
        os.makedirs(save_dir, exist_ok=True)

        benchmark.run(
            print_data=True,
            show_plots=True,
            save_path=save_dir,
            N=N,
            K=K,
        )

    print("Benchmark finished!")
