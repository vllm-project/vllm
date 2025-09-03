#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    silu_mul_fp8_quant_deep_gemm_cuda,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used


@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(
            input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        y_s = tl.maximum(tl.max(tl.abs(y)), eps) / fp8_max
        if use_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def silu_mul_fp8_quant_deep_gemm_triton(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens,
    group_size: int = 128,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales

    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.

    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = H // group_size
    assert H % group_size == 0, "H must be divisible by group_size"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E, (
        "tokens_per_expert must be shape (E,)"
    )
    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    # allocate outputs
    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    # strides (elements)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    # desired scale strides (elements): (T*G, 1, T)
    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    y_s = torch.empty_strided(
        (E, T, G),
        (stride_ys_e, stride_ys_t, stride_ys_g),
        dtype=torch.float32,
        device=y.device,
    )

    stride_cnt_e = tokens_per_expert.stride()[0]

    # Static grid over experts and H-groups.
    # A loop inside the kernel handles the token dim
    grid = (E * G,)

    f_info = torch.finfo(fp8_dtype)
    fp8_max = f_info.max
    fp8_min = f_info.min

    _silu_mul_fp8_quant_deep_gemm[grid](
        y,
        y_q,
        y_s,
        tokens_per_expert,
        H,
        group_size,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yq_e,
        stride_yq_t,
        stride_yq_h,
        stride_ys_e,
        stride_ys_t,
        stride_ys_g,
        stride_cnt_e,
        eps,
        fp8_min,
        fp8_max,
        is_deep_gemm_e8m0_used(),
        BLOCK=group_size,
        NUM_STAGES=4,
        num_warps=1,
    )

    return y_q, y_s


def benchmark(k, E, T, H, num_parallel_tokens, G=128, runs=100):
    current_platform.seed_everything(42)
    y = torch.randn((E, T, 2 * H), dtype=torch.bfloat16, device="cuda")
    tokens_per_expert = torch.randint(
        T // 2, T, size=(E,), dtype=torch.int32, device="cuda"
    )

    # Warmup
    for _ in range(20):
        k(y, tokens_per_expert, num_parallel_tokens=num_parallel_tokens, group_size=G)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        k(y, tokens_per_expert, num_parallel_tokens=num_parallel_tokens, group_size=G)
    torch.cuda.synchronize()

    avg_time = (time.perf_counter() - start) / runs * 1000

    # Calculate actual work done (only count valid tokens)
    actual_tokens = tokens_per_expert.sum().item()
    actual_elements = actual_tokens * H

    # GFLOPS: operations per element = exp + 3 muls + 1 div + quantization ops ≈ 8 ops
    ops_per_element = 8
    total_ops = actual_elements * ops_per_element
    gflops = total_ops / (avg_time / 1000) / 1e9

    # Memory bandwidth: bfloat16 inputs (2 bytes), fp8 output (1 byte), scales (4 bytes)
    input_bytes = actual_tokens * 2 * H * 2  # 2*H bfloat16 inputs
    output_bytes = actual_tokens * H * 1  # H fp8 outputs
    scale_bytes = actual_tokens * (H // G) * 4  # scales in float32
    total_bytes = input_bytes + output_bytes + scale_bytes
    memory_bw = total_bytes / (avg_time / 1000) / 1e9

    return avg_time, gflops, memory_bw


def benchmark_full():
    configs = [
        (8, 8, 7168),
        (8, 16, 7168),
        (8, 32, 7168),
        (8, 64, 7168),
        (8, 128, 7168),
        (8, 256, 7168),
        (8, 512, 7168),
        (8, 1024, 7168),
        (32, 8, 7168),
        (32, 16, 7168),
        (32, 32, 7168),
        (32, 64, 7168),
        (32, 128, 7168),
        (32, 256, 7168),
        (32, 512, 7168),
        (32, 1024, 7168),
        (256, 8, 7168),
        (256, 16, 7168),
        (256, 32, 7168),
        (256, 64, 7168),
        (256, 128, 7168),
        (256, 256, 7168),
        (256, 512, 7168),
        (256, 1024, 7168),
    ]

    num_parallel_tokens = 16

    print(f"GPU: {torch.cuda.get_device_name()} CUDA Kernel")
    print(f"{'Config':<20} {'Time(ms)':<10} {'GFLOPS':<10} {'GB/s':<10}")
    print("-" * 50)

    for E, T, H in configs:
        time_ms, gflops, gbps = benchmark(
            silu_mul_fp8_quant_deep_gemm_cuda, E, T, H, num_parallel_tokens
        )
        print(f"E={E:3d},T={T:4d},H={H:4d} {time_ms:8.3f} {gflops:8.1f} {gbps:8.1f}")

    print(f"GPU: {torch.cuda.get_device_name()} Baseline")
    print(f"{'Config':<20} {'Time(ms)':<10} {'GFLOPS':<10} {'GB/s':<10}")
    print("-" * 50)

    for E, T, H in configs:
        time_ms, gflops, gbps = benchmark(
            silu_mul_fp8_quant_deep_gemm_triton, E, T, H, num_parallel_tokens
        )
        print(f"E={E:3d},T={T:4d},H={H:4d} {time_ms:8.3f} {gflops:8.1f} {gbps:8.1f}")


benchmark_full()
