#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    silu_mul_fp8_quant_deep_gemm,
)
from vllm.platforms import current_platform


def benchmark(E, T, H, G=128, runs=50):
    current_platform.seed_everything(42)
    y = torch.randn((E, T, 2 * H), dtype=torch.bfloat16, device="cuda")
    tokens_per_expert = torch.randint(
        T // 2, T, size=(E,), dtype=torch.int32, device="cuda"
    )

    # Warmup
    for _ in range(10):
        silu_mul_fp8_quant_deep_gemm(y, tokens_per_expert, group_size=G)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        silu_mul_fp8_quant_deep_gemm(y, tokens_per_expert, group_size=G)
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


configs = [
    (8, 32, 1024),
    (16, 64, 2048),
    (32, 128, 4096),
    # DeepSeekV3 Configs
    (256, 16, 7168),
    (256, 32, 7168),
    (256, 64, 7168),
    (256, 128, 7168),
    (256, 256, 7168),
    (256, 512, 7168),
    (256, 1024, 7168),
]

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"{'Config':<20} {'Time(ms)':<10} {'GFLOPS':<10} {'GB/s':<10}")
print("-" * 50)

for E, T, H in configs:
    try:
        time_ms, gflops, gbps = benchmark(E, T, H)
        print(f"E={E:3d},T={T:4d},H={H:4d} {time_ms:8.3f} {gflops:8.1f} {gbps:8.1f}")
    except Exception:
        print(f"E={E:3d},T={T:4d},H={H:4d} FAILED")
