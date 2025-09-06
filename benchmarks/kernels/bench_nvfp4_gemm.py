# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import copy
import itertools

import torch
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton

import triton.language as tl

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 requires compute capability of 10.0 (Blackwell)")


FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

PROVIDER_CFGS = {
    "torch-bf16": dict(enabled=True),
    "nvfp4": dict(no_a_quant=False, enabled=True),
    "nvfp4-noquant": dict(no_a_quant=True, enabled=True),
    "triton-fp4": dict(enabled=True),
}

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]


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

@triton.jit
def block_scale_fp4_matmul(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr, PACK_ALONG_K: tl.constexpr):  #
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    PACKING_ALONG_M_N: tl.constexpr = 1 if PACK_ALONG_K else 2
    offs_am_packed = (pid_m * (BLOCK_M // PACKING_ALONG_M_N) + tl.arange(0, BLOCK_M // PACKING_ALONG_M_N))
    offs_bn_packed = (pid_n * (BLOCK_N // PACKING_ALONG_M_N) + tl.arange(0, BLOCK_N // PACKING_ALONG_M_N))
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2 if PACK_ALONG_K else BLOCK_K

    # Two e2m1 values per K
    offs_k = tl.arange(0, BLOCK_K_PACKED)
    offs_scale_k = tl.arange(0, BLOCK_K // VEC_SIZE)
    if a_scale is not None:
        a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    if b_scale is not None:
        b_scale_ptr = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am_packed[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn_packed[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if a_scale is not None:
            scale_a = tl.load(a_scale_ptr)
        else:
            scale_a = None
        if b_scale is not None:
            scale_b = tl.load(b_scale_ptr)
        else:
            scale_b = None
        accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator, lhs_k_pack=PACK_ALONG_K,
                                    rhs_k_pack=PACK_ALONG_K)
        a_ptrs += (BLOCK_K_PACKED) * stride_ak
        b_ptrs += (BLOCK_K_PACKED) * stride_bk
        if a_scale is not None:
            a_scale_ptr += BLOCK_K // VEC_SIZE
        if b_scale is not None:
            b_scale_ptr += BLOCK_K // VEC_SIZE
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)

def build_triton_fp4_runner(a: torch.Tensor, b: torch.Tensor, dtype, device: str):
    """
    Launches the Triton FP4 matmul using the existing block_scale_fp4_matmul kernel.
    - Quantizes B once (weights) and A on-the-fly per call.
    - Accumulates into BF16 (dtype).
    """

    # weights
    b_fp4, scale_b_fp4, b_global_scale = _quant_weight_nvfp4(b, device)

    b_fp4 = b_fp4.T.contiguous()  

    # activation
    a_amax = torch.abs(a).max().to(torch.float32)
    a_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax

    M, K = a.shape
    N, _ = b.shape
    c = torch.empty((M, N), device=device, dtype=dtype)

    VEC_SIZE     = 16           
    BLOCK_M      = 128
    BLOCK_N      = 128
    BLOCK_K      = 128
    NUM_STAGES   = 3
    PACK_ALONG_K = 1           

    # row major
    stride_am, stride_ak = a.stride()      
    stride_bk, stride_bn = b_fp4.stride()
    stride_cm, stride_cn = c.stride()      

    stride_scale = K // VEC_SIZE

    def run():
        a_fp4, scale_a_fp4 = ops.scaled_fp4_quant(a, a_global_scale)

        stride_am, stride_ak = a_fp4.stride()
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        block_scale_fp4_matmul[grid](
            # pointers
            a_fp4, b_fp4, c,
            scale_a_fp4, scale_b_fp4,

            # sizes
            M, N, K,

            # scale stride
            stride_scale,

            # tensor strides
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,

            # meta
            VEC_SIZE=VEC_SIZE,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_STAGES=NUM_STAGES, PACK_ALONG_K=PACK_ALONG_K,
        )
        return c

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
    elif provider == "triton-fp4":                    
        run_triton = build_triton_fp4_runner(a, b, dtype, device)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: run_triton(), quantiles=quantiles
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
