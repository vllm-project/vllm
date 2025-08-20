#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    silu_mul_fp8_quant_deep_gemm,
)
from vllm.platforms import current_platform


def benchmark(E, T, H, G=128, runs=10):
    current_platform.seed_everything(42)
    y = torch.randn((E, T, 2 * H), dtype=torch.float32, device="cuda")
    tokens_per_expert = torch.randint(
        T // 2, T, size=(E,), dtype=torch.int32, device="cuda"
    )

    # Warmup
    for _ in range(3):
        silu_mul_fp8_quant_deep_gemm(y, tokens_per_expert, group_size=G)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        silu_mul_fp8_quant_deep_gemm(y, tokens_per_expert, group_size=G)
    torch.cuda.synchronize()

    avg_time = (time.perf_counter() - start) / runs * 1000
    throughput = E * T * H / (avg_time / 1000) / 1e9

    # Memory bandwidth: Input E*T*2*H*4 + Output E*T*H*1 + E*T*(H/G)*4 bytes
    input_bytes = E * T * 2 * H * 4
    output_bytes = E * T * H * 1 + E * T * (H // G) * 4
    total_bytes = input_bytes + output_bytes
    memory_bw = total_bytes / (avg_time / 1000) / 1e9

    return avg_time, throughput, memory_bw


configs = [
    (8, 32, 1024),
    (16, 64, 2048),
    (32, 128, 4096),
    # DeepSeekV3 Configs
    (256, 1, 7168),
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
