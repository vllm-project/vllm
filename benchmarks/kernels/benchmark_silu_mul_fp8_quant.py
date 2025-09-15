# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

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
strategies = ["uniform", "max_t", "first_t"]


def benchmark(
    kernel: Callable,
    E: int,
    T: int,
    H: int,
    total_tokens: int,
    num_parallel_tokens: int = 64,
    G: int = 128,
    runs: int = 200,
    num_warmups: int = 20,
    gen_strategy: str = "default",
    iterations_per_run: int = 20,
):
    def generate_data(seed_offset=0):
        """Generate input data with given seed offset"""
        current_platform.seed_everything(42 + seed_offset)
        y = torch.rand((E, T, 2 * H), dtype=torch.bfloat16, device="cuda").contiguous()

        if gen_strategy == "uniform":
            r = torch.rand(size=(E,), device="cuda")
            r /= r.sum()
            r *= total_tokens
            tokens_per_expert = r.int()
            tokens_per_expert = torch.minimum(
                tokens_per_expert,
                torch.ones((E,), device=r.device, dtype=torch.int) * T,
            )
        elif gen_strategy == "max_t":
            tokens_per_expert = torch.empty(size=(E,), dtype=torch.int32, device="cuda")
            tokens_per_expert.fill_(total_tokens / E)
        elif gen_strategy == "first_t":
            tokens_per_expert = torch.zeros(size=(E,), dtype=torch.int32, device="cuda")
            tokens_per_expert[0] = min(T, total_tokens)
        else:
            raise ValueError(f"Unknown generation strategy: {gen_strategy}")
        return y, tokens_per_expert

    dataset_count = 4
    # Pre-generate different input matrices for each iteration to avoid cache effects
    data_sets = [generate_data(i) for i in range(dataset_count)]

    # Warmup
    y, tokens_per_expert = data_sets[0]
    for _ in range(num_warmups):
        kernel(
            y, tokens_per_expert, num_parallel_tokens=num_parallel_tokens, group_size=G
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark
    latencies: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()

        start_event.record()
        for i in range(iterations_per_run):
            y, tokens_per_expert = data_sets[i % dataset_count]
            kernel(
                y,
                tokens_per_expert,
                num_parallel_tokens=num_parallel_tokens,
                group_size=G,
            )
        end_event.record()
        end_event.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        per_iter_time_ms = total_time_ms / iterations_per_run
        latencies.append(per_iter_time_ms)

    # Use median instead of average for better outlier handling
    median_time_ms = np.median(latencies)
    median_time_s = median_time_ms / 1000

    # Calculate actual work done (using first dataset for consistency)
    _, tokens_per_expert = data_sets[0]
    actual_tokens = tokens_per_expert.sum().item()
    actual_elements = actual_tokens * H

    # GFLOPS: operations per element = exp + 3 muls + 1 div + quantization ops ≈ 8 ops
    ops_per_element = 8
    total_ops = actual_elements * ops_per_element
    gflops = total_ops / median_time_s / 1e9

    # Memory bandwidth: bfloat16 inputs (2 bytes), fp8 output (1 byte), scales (4 bytes)
    input_bytes = actual_tokens * 2 * H * 2  # 2*H bfloat16 inputs
    output_bytes = actual_tokens * H * 1  # H fp8 outputs
    scale_bytes = actual_tokens * (H // G) * 4  # scales in float32
    total_bytes = input_bytes + output_bytes + scale_bytes
    memory_bw = total_bytes / median_time_s / 1e9

    HOPPER_BANDWIDTH_TBPS = 3.35
    return (
        median_time_ms,
        gflops,
        memory_bw,
        (memory_bw / (HOPPER_BANDWIDTH_TBPS * 1024)) * 100,
    )


def create_comparison_plot(
    ratio, cuda_times, baseline_times, config_labels, strategy_name, id
):
    """Create a comparison plot for a specific generation strategy"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

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
    return fig, ax


def create_combined_plot(all_results):
    """Create a combined plot with all strategies in one PNG"""
    num_strategies = len(all_results)
    fig, axes = plt.subplots(num_strategies, 1, figsize=(20, 6 * num_strategies))

    if num_strategies == 1:
        axes = [axes]

    for idx, (
        strategy_name,
        ratio,
        cuda_times,
        baseline_times,
        config_labels,
    ) in enumerate(all_results):
        ax = axes[idx]

        # Configure x-axis positions
        x = np.arange(len(config_labels))
        width = 0.35

        # Execution Time plot (lower is better)
        ax.bar(
            x - width / 2,
            cuda_times,
            width,
            label="CUDA Kernel",
            alpha=0.8,
            color="blue",
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
    filename = "../../silu_bench/silu_benchmark_combined.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return filename


outer_dim = 7168
configs = [
    # DeepSeekV3 Configs
    (8, 1024, 7168),
    # DeepSeekV3 Configs
    (32, 1024, 7168),
    # DeepSeekV3 Configs
    (256, 1024, 7168),
]

runs = 100
num_warmups = 20

strategy_descriptions = {
    "uniform": "Uniform Random",
    "max_t": "Even Assignment",
    "first_t": "experts[0] = T, experts[1:] = 0",
}

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Testing strategies: {', '.join(strategies)}")
print(f"Configurations: {len(configs)} configs")

all_results = []

# Run benchmarks for each strategy
for id, strategy in enumerate(strategies):
    print(f"\n{'=' * 60}")
    print(f"Testing strategy: {strategy_descriptions[strategy]}")
    print(f"{'=' * 60}")

    # Collect benchmark data for both algorithms
    config_labels = []
    config_x_axis = []
    all_cuda_results = []
    all_baseline_results = []
    all_ratios = []

    for E, T, H in configs:
        total_tokens_config = [8 * E, 16 * E, 32 * E, 64 * E, 128 * E, 256 * E]
        config_x_axis.append(total_tokens_config)

        cuda_results = []
        baseline_results = []
        ratios = []

        for total_tokens in total_tokens_config:
            config_label = f"E={E},T={T},H={H},TT={total_tokens}"
            config_labels.append(config_label)

            # CUDA kernel results
            time_ms_cuda, gflops, gbps, perc = benchmark(
                silu_mul_fp8_quant_deep_gemm_cuda,
                E,
                T,
                H,
                total_tokens,
                runs=runs,
                num_warmups=num_warmups,
                gen_strategy=strategy,
            )
            cuda_results.append((time_ms_cuda, gflops, gbps, perc))

            # Baseline results
            time_ms_triton, gflops, gbps, perc = benchmark(
                silu_mul_fp8_quant_deep_gemm_triton,
                E,
                T,
                H,
                total_tokens,
                runs=runs,
                num_warmups=num_warmups,
                gen_strategy=strategy,
            )
            baseline_results.append((time_ms_triton, gflops, gbps, perc))
            ratios.append(time_ms_triton / time_ms_cuda)

            print(f"Completed: {config_label}")
        all_cuda_results.append(cuda_results)
        all_baseline_results.append(baseline_results)
        all_ratios.append(ratios)

    # Store results for combined plotting
    all_results.append(
        (
            strategy_descriptions[strategy],
            all_ratios,
            all_cuda_results,
            all_baseline_results,
            config_labels,
            config_x_axis,
        )
    )

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


def create_total_tokens_plot(all_results):
    num_strategies = len(all_results)
    num_configs = len(configs)

    # Create side-by-side subplots: 2 columns for speedup and bandwidth percentage
    fig, axs = plt.subplots(
        num_strategies, num_configs * 2, figsize=(28, 6 * num_strategies)
    )

    # Add main title to the entire figure
    fig.suptitle(
        "Performance Analysis: Speedup vs Bandwidth Utilization (Triton & CUDA)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Handle single strategy case
    if num_strategies == 1:
        axs = axs.reshape(1, -1)

    # Handle single config case
    if num_configs == 1:
        axs = axs.reshape(-1, 2)

    for strategy_idx, result in enumerate(all_results):
        (
            strategy_name,
            all_ratios,
            all_cuda_results,
            all_baseline_results,
            config_labels,
            config_x_axis,
        ) = result

        for config_idx in range(num_configs):
            # Speedup plot (left column)
            ax_speedup = axs[strategy_idx, config_idx * 2]
            # Bandwidth plot (right column)
            ax_bandwidth = axs[strategy_idx, config_idx * 2 + 1]

            E, T, H = configs[config_idx]
            ratios = all_ratios[config_idx]
            total_tokens_values = config_x_axis[config_idx]

            # Extract CUDA and Triton bandwidth percentages
            cuda_bandwidth_percentages = [
                result[3] for result in all_cuda_results[config_idx]
            ]
            triton_bandwidth_percentages = [
                result[3] for result in all_baseline_results[config_idx]
            ]

            # Plot speedup ratios vs total tokens (left plot)
            ax_speedup.plot(
                total_tokens_values, ratios, "bo-", linewidth=3, markersize=8
            )
            ax_speedup.set_title(
                f"{strategy_name}\nSpeedup (CUDA/Triton)\nE={E}, T={T}, H={H}",
                fontsize=12,
                fontweight="bold",
            )
            ax_speedup.set_xlabel("Total Tokens", fontweight="bold", fontsize=11)
            ax_speedup.set_ylabel("Speedup Ratio", fontweight="bold", fontsize=11)
            ax_speedup.grid(True, alpha=0.3)

            ax_bandwidth.plot(
                total_tokens_values,
                cuda_bandwidth_percentages,
                "ro-",
                linewidth=3,
                markersize=8,
                label="CUDA",
            )
            ax_bandwidth.plot(
                total_tokens_values,
                triton_bandwidth_percentages,
                "go-",
                linewidth=3,
                markersize=8,
                label="Triton",
            )
            ax_bandwidth.set_title(
                f"{strategy_name}\nBandwidth Utilization (Hopper)\nE={E}, T={T}, H={H}",
                fontsize=12,
                fontweight="bold",
            )
            ax_bandwidth.set_xlabel("Total Tokens", fontweight="bold", fontsize=11)
            ax_bandwidth.set_ylabel(
                "% of Peak Bandwidth", fontweight="bold", fontsize=11
            )
            ax_bandwidth.legend(prop={"weight": "bold"})
            ax_bandwidth.grid(True, alpha=0.3)

            # Format x-axis labels for both plots
            for ax in [ax_speedup, ax_bandwidth]:
                ax.set_xticks(total_tokens_values)
                ax.set_xticklabels(
                    [
                        f"{tt // 1000}K" if tt >= 1000 else str(tt)
                        for tt in total_tokens_values
                    ],
                    fontweight="bold",
                )
                # Make tick labels bold
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight("bold")

            # Add value labels on speedup points
            for x, y in zip(total_tokens_values, ratios):
                ax_speedup.annotate(
                    f"{y:.2f}x",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

            # Add value labels on CUDA bandwidth points
            for x, y in zip(total_tokens_values, cuda_bandwidth_percentages):
                ax_bandwidth.annotate(
                    f"{y:.1f}%",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.3),
                )

            # Add value labels on Triton bandwidth points
            for x, y in zip(total_tokens_values, triton_bandwidth_percentages):
                ax_bandwidth.annotate(
                    f"{y:.1f}%",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="green", alpha=0.3),
                )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for main title
    filename = "silu_benchmark_total_tokens.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return filename


# Create combined plot with all strategies
combined_plot_filename = create_total_tokens_plot(all_results)

print(f"\n{'=' * 60}")
print("Benchmark Complete!")
print(f"Generated combined plot: {combined_plot_filename}")
print(f"{'=' * 60}")
