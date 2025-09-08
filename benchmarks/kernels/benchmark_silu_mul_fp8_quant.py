# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import matplotlib.pyplot as plt
import numpy as np
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
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

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
    G = (H + group_size - 1) // group_size
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


# Parse generation strategies
strategies = ["max_t", "uniform", "first_t"]


def benchmark(
    k,
    E,
    T,
    H,
    num_parallel_tokens=64,
    G=128,
    runs=100,
    num_warmups=20,
    gen_strategy="default",
):
    current_platform.seed_everything(42)

    y = torch.rand((E, T, 2 * H), dtype=torch.bfloat16, device="cuda").contiguous()
    # Different random generation strategies
    if gen_strategy == "uniform":
        tokens_per_expert = torch.randint(
            int(T * 0.7), T, size=(E,), dtype=torch.int32, device="cuda"
        )
    elif gen_strategy == "max_t":
        tokens_per_expert = torch.ones(size=(E,), dtype=torch.int32, device="cuda") * T
    elif gen_strategy == "first_t":
        tokens_per_expert = torch.zeros(size=(E,), dtype=torch.int32, device="cuda") * 0
        tokens_per_expert[0] = T
    elif gen_strategy == "sorted":
        tokens_per_expert = torch.randint(
            0, T, size=(E,), dtype=torch.int32, device="cuda"
        )
        tokens_per_expert, _ = torch.sort(tokens_per_expert)
    else:
        raise ValueError(f"Unknown generation strategy: {gen_strategy}")

    # Warmup
    for _ in range(num_warmups):
        k(y, tokens_per_expert, num_parallel_tokens=num_parallel_tokens, group_size=G)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark
    latencies: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()

        start_event.record()
        k(y, tokens_per_expert, num_parallel_tokens=num_parallel_tokens, group_size=G)
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    avg_time_ms = sum(latencies) / runs
    avg_time_s = avg_time_ms / 1000

    # Calculate actual work done (only count valid tokens)
    actual_tokens = tokens_per_expert.sum().item()
    actual_elements = actual_tokens * H

    # GFLOPS: operations per element = exp + 3 muls + 1 div + quantization ops ≈ 8 ops
    ops_per_element = 8
    total_ops = actual_elements * ops_per_element
    gflops = total_ops / avg_time_s / 1e9

    # Memory bandwidth: bfloat16 inputs (2 bytes), fp8 output (1 byte), scales (4 bytes)
    input_bytes = actual_tokens * 2 * H * 2  # 2*H bfloat16 inputs
    output_bytes = actual_tokens * H * 1  # H fp8 outputs
    scale_bytes = actual_tokens * (H // G) * 4  # scales in float32
    total_bytes = input_bytes + output_bytes + scale_bytes
    memory_bw = total_bytes / avg_time_s / 1e9

    return avg_time_ms, gflops, memory_bw, (memory_bw / (3.35 * 1024)) * 100


def create_comparison_plot(
    ratio, cuda_times, baseline_times, config_labels, strategy_name, id
):
    """Create a comparison plot for a specific generation strategy"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Configure x-axis positions
    x = np.arange(len(config_labels))
    width = 0.35

    # Execution Time plot (lower is better)
    ax.bar(
        x - width / 2, cuda_times, width, label="CUDA Kernel", alpha=0.8, color="blue"
    )
    ax.bar(
        x + width / 2,
        baseline_times,
        width,
        label="Baseline",
        alpha=0.8,
        color="orange",
    )

    # Add speedup labels over each bar pair
    for i in range(len(x)):
        speedup = ratio[i]
        max_height = max(cuda_times[i], baseline_times[i])
        ax.text(
            x[i],
            max_height + max_height * 0.02,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("% Utilization")
    ax.set_title(
        f"Memory Bandwidth Utilization (%) - {strategy_name}\n(Higher is Better)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"../silu_bench/silu_benchmark_{id}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return filename


outer_dim = 7168
configs = [
    # DeepSeekV3 Configs
    (9, 8, 7168),
    (9, 16, 7168),
    (9, 32, 7168),
    (9, 64, 7168),
    (9, 128, 7168),
    (9, 256, 7168),
    (9, 512, 7168),
    (9, 1024, 7168),
    # DeepSeekV3 Configs
    (32, 8, 7168),
    (32, 16, 7168),
    (32, 32, 7168),
    (32, 64, 7168),
    (32, 128, 7168),
    (32, 256, 7168),
    (32, 512, 7168),
    (32, 1024, 7168),
    # DeepSeekV3 Configs
    (256, 8, 7168),
    (256, 16, 7168),
    (256, 32, 7168),
    (256, 64, 7168),
    (256, 128, 7168),
    (256, 256, 7168),
    (256, 512, 7168),
    (256, 1024, 7168),
]

runs = 100
num_warmups = 20

strategy_descriptions = {
    "uniform": "experts = torch.randint(int(T * 0.7), T, size=(E,))",
    "max_t": "experts[:] = T",
    "first_t": "experts[0] = T, experts[1:] = 0",
}

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Testing strategies: {', '.join(strategies)}")
print(f"Configurations: {len(configs)} configs")

generated_plots = []

# Run benchmarks for each strategy
for id, strategy in enumerate(strategies):
    print(f"\n{'=' * 60}")
    print(f"Testing strategy: {strategy_descriptions[strategy]}")
    print(f"{'=' * 60}")

    # Collect benchmark data for both algorithms
    cuda_results = []
    baseline_results = []
    config_labels = []

    for E, T, H in configs:
        config_label = f"E={E},T={T},H={H}"
        config_labels.append(config_label)

        # CUDA kernel results
        time_ms, gflops, gbps, perc = benchmark(
            silu_mul_fp8_quant_deep_gemm_cuda,
            E,
            T,
            H,
            runs=runs,
            num_warmups=num_warmups,
            gen_strategy=strategy,
        )
        cuda_results.append((time_ms, gflops, gbps, perc))

        # Baseline results
        time_ms, gflops, gbps, perc = benchmark(
            silu_mul_fp8_quant_deep_gemm_triton,
            E,
            T,
            H,
            runs=runs,
            num_warmups=num_warmups,
            gen_strategy=strategy,
        )
        baseline_results.append((time_ms, gflops, gbps, perc))

        print(f"Completed: {config_label}")

    # Extract data for plotting
    cuda_times = [r[3] for r in cuda_results]
    baseline_times = [r[3] for r in baseline_results]

    ratio = [
        baseline_results[i][0] / cuda_results[i][0] for i in range(len(config_labels))
    ]
    # Create comparison plot for this strategy
    plot_filename = create_comparison_plot(
        ratio,
        cuda_times,
        baseline_times,
        config_labels,
        strategy_descriptions[strategy],
        id,
    )
    generated_plots.append(plot_filename)

    # Print summary table for this strategy
    print(f"\nSummary Table - {strategy_descriptions[strategy]}:")
    print(f"{'Config':<20} {'CUDA Time(ms)':<12} {'Base Time(ms)':<12} {'Speedup':<8}")
    print("-" * 60)

    for i, (E, T, H) in enumerate(configs):
        speedup = baseline_results[i][0] / cuda_results[i][0]
        config_label = f"E={E:3d},T={T:4d},H={H:4d}"
        print(
            f"{config_label:<20} {cuda_results[i][0]:8.5f} "
            f"{baseline_results[i][0]:8.5f} {speedup:6.2f}x"
        )

print(f"\n{'=' * 60}")
print("Benchmark Complete!")
print(f"Generated plots: {', '.join(generated_plots)}")
print(f"{'=' * 60}")
