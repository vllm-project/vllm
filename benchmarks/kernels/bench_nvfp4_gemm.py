# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import copy
import itertools

import torch
import triton
import triton.language as tl
from typing import Optional, List, Dict
import pandas as pd
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
}

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]


def benchmark_nvfp4_gemm_enhanced(
    m: int,
    n: int,
    k: int,
    use_cutlass: bool = True,
    use_triton: bool = True,
    dtype: torch.dtype = torch.float16
) -> List[Dict]:
    results = []
    
    device = torch.device('cuda')
    compute_capability = torch.cuda.get_device_capability(device)
    sm_version = compute_capability[0] * 10 + compute_capability[1]
    
    print(f"Device SM version: {sm_version}")
    
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    
    if use_cutlass and sm_version == 100:
        try:
            from vllm.experimental.kernels import cutlass_fp8_gemm
            
            for _ in range(10):
                out_cutlass = cutlass_fp8_gemm(a, b)
            
            torch.cuda.synchronize()
            
            import time
            num_iterations = 100
            start = time.time()
            
            for _ in range(num_iterations):
                out_cutlass = cutlass_fp8_gemm(a, b)
            
            torch.cuda.synchronize()
            end = time.time()
            
            elapsed_ms = (end - start) * 1000 / num_iterations
            flops = 2 * m * n * k
            tflops = flops / (elapsed_ms / 1000) / 1e12
            
            results.append({
                'implementation': 'CUTLASS',
                'elapsed_ms': elapsed_ms,
                'tflops': tflops,
                'm': m,
                'n': n,
                'k': k,
                'sm_version': sm_version
            })
            
        except ImportError:
            print("CUTLASS implementation not available")
    
    if use_triton:
        triton_result = benchmark_triton_nvfp4_gemm(m, n, k, dtype)
        triton_result.update({
            'implementation': 'Triton',
            'm': m,
            'n': n,
            'k': k,
            'sm_version': sm_version
        })
        results.append(triton_result)
    
    for _ in range(10):
        out_ref = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    
    import time
    num_iterations = 100
    start = time.time()
    
    for _ in range(num_iterations):
        out_ref = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed_ms = (end - start) * 1000 / num_iterations
    flops = 2 * m * n * k
    tflops = flops / (elapsed_ms / 1000) / 1e12
    
    results.append({
        'implementation': 'PyTorch (FP16)',
        'elapsed_ms': elapsed_ms,
        'tflops': tflops,
        'm': m,
        'n': n,
        'k': k,
        'sm_version': sm_version
    })
    
    return results


def _quant_weight_nvfp4(b: torch.Tensor, device: str):
    # Compute global scale for weight
    b_amax = torch.abs(b).max().to(torch.float32)
    b_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax
    b_fp4, scale_b_fp4 = ops.scaled_fp4_quant(b, b_global_scale)
    return b_fp4, scale_b_fp4, b_global_scale


def build_nvfp4_runner(cfg, a, b, dtype, device):
    b_fp4, scale_b_fp4, b_global_scale = _quant_weight_nvfp4(b, device)

    # Compute global scale for activation
    # NOTE: This is generally provided ahead-of-time by the model checkpoint.
    a_amax = torch.abs(a).max().to(torch.float32)
    a_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax

    # Alpha for the GEMM operation
    alpha = 1.0 / (a_global_scale * b_global_scale)

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
        benchmark.run(
            print_data=True,
            show_plots=True,
            save_path=f"bench_nvfp4_res_n{N}_k{K}",
            N=N,
            K=K,
        )

    print("Benchmark finished!")
