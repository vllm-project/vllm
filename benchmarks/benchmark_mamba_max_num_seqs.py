# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark to find the optimal --max-num-seqs for Mamba/hybrid models.

This script benchmarks the SSM kernel at various batch sizes and recommends
an optimal --max-num-seqs value based on where bandwidth reaches near-peak.

Algorithm:
1. Find the smallest batch size that achieves 99% of maximum observed bandwidth
2. Apply 2x headroom for max throughput
Tested on NVIDIA B200 with Nemotron-H models.

Example:
    python benchmark_mamba_max_num_seqs.py \
        --model nvidia/Nemotron-H-8B-Base-8K
"""

import argparse
from typing import NamedTuple

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

DEFAULT_BATCH_SIZES = [
    32,
    64,
    96,
    128,
    192,
    224,
    256,
    288,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
]


class MambaConfig(NamedTuple):
    num_heads: int
    head_dim: int
    state_size: int
    num_mamba_layers: int
    state_dtype: torch.dtype
    ngroups: int = 1


class BenchmarkResult(NamedTuple):
    batch_size: int
    time_ms: float
    effective_bandwidth_tb_s: float


def get_config_from_model(
    model_path: str,
    state_dtype: torch.dtype,
    trust_remote_code: bool = True,
) -> MambaConfig:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    required = ["mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups"]
    missing = [attr for attr in required if getattr(hf_config, attr, None) is None]
    if missing:
        raise ValueError(f"Model config missing required attributes: {missing}")

    hybrid_pattern = getattr(hf_config, "hybrid_override_pattern", None)
    if not hybrid_pattern:
        raise ValueError("Model config missing 'hybrid_override_pattern'")

    num_mamba_layers = hybrid_pattern.count("M")
    print(f"Model: {model_path}")
    print(
        f"  heads={hf_config.mamba_num_heads}, head_dim={hf_config.mamba_head_dim}, "
        f"state={hf_config.ssm_state_size}, groups={hf_config.n_groups}, "
        f"mamba_layers={num_mamba_layers}"
    )

    return MambaConfig(
        num_heads=hf_config.mamba_num_heads,
        head_dim=hf_config.mamba_head_dim,
        state_size=hf_config.ssm_state_size,
        num_mamba_layers=num_mamba_layers,
        state_dtype=state_dtype,
        ngroups=hf_config.n_groups,
    )


def benchmark_ssm_kernel(
    config: MambaConfig,
    batch_size: int,
    compute_dtype: torch.dtype = torch.bfloat16,
    num_iterations: int = 50,
    warmup_iterations: int = 10,
) -> BenchmarkResult:
    device = torch.device("cuda:0")
    nheads, dim, dstate = config.num_heads, config.head_dim, config.state_size
    num_layers = config.num_mamba_layers

    # Create tensors matching Mamba2 decode phase shapes.
    # These shapes follow the "with heads" variant used by MambaMixer2.
    # Reference: tests/kernels/mamba/test_mamba_ssm.py
    #   - see test_selective_state_update_with_heads_with_batch_indices()
    #   - state: (batch, nheads, headdim, dstate) - uses state_dtype
    #   - x, dt, z: (batch, nheads, headdim) - uses compute_dtype
    #   - A: (nheads, headdim, dstate) - always float32 for precision
    #   - B, C: (batch, ngroups, dstate) - uses compute_dtype
    #   - D, dt_bias: (nheads, headdim) - uses compute_dtype
    state = torch.randn(
        batch_size, nheads, dim, dstate, device=device, dtype=config.state_dtype
    )
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=compute_dtype)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=compute_dtype)
    A = torch.randn(nheads, dim, dstate, device=device, dtype=torch.float32) * -0.1
    B = torch.randn(
        batch_size, config.ngroups, dstate, device=device, dtype=compute_dtype
    )
    C = torch.randn(
        batch_size, config.ngroups, dstate, device=device, dtype=compute_dtype
    )
    D = torch.randn(nheads, dim, device=device, dtype=compute_dtype)
    z = torch.randn(batch_size, nheads, dim, device=device, dtype=compute_dtype)
    dt_bias = torch.randn(nheads, dim, device=device, dtype=compute_dtype)
    out = torch.empty_like(x)

    def run_layers():
        for _ in range(num_layers):
            selective_state_update(
                state=state,
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=True,
                out=out,
            )

    # Warmup
    for _ in range(warmup_iterations):
        run_layers()
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    for i in range(num_iterations):
        start_events[i].record()
        run_layers()
        end_events[i].record()

    torch.cuda.synchronize()
    avg_time_ms = (
        sum(s.elapsed_time(e) for s, e in zip(start_events, end_events))
        / num_iterations
    )

    # Calculate effective memory bandwidth.
    # The SSM state is the dominant memory traffic - it's read and written each layer.
    # state_size = batch * nheads * headdim * dstate * dtype_bytes
    # total_traffic = state_size * 2 (read + write) * num_layers
    dtype_bytes = 4 if config.state_dtype == torch.float32 else 2
    state_traffic_gb = (
        batch_size * nheads * dim * dstate * dtype_bytes * num_layers * 2 / 1e9
    )
    effective_bandwidth_tb_s = state_traffic_gb / (avg_time_ms / 1000) / 1000

    return BenchmarkResult(batch_size, avg_time_ms, effective_bandwidth_tb_s)


def find_optimal_batch_size(
    config: MambaConfig,
    batch_sizes: list[int] | None = None,
    tp_size: int = 1,
    compute_dtype: torch.dtype = torch.bfloat16,
    bw_threshold: float = 0.005,
    near_peak_ratio: float = 0.99,
    throughput_headroom: float = 2.0,
) -> tuple[int, int, list[BenchmarkResult]]:
    """
    Find the optimal batch size for serving throughput.

    Uses bandwidth plateau detection: finds where bandwidth reaches near-peak
    (99% of max), then applies 2x headroom for overall system efficiency.
    This is more stable than threshold-based saturation detection.
    """
    if tp_size > 1:
        config = config._replace(num_heads=config.num_heads // tp_size)
        print(f"\nAdjusted for TP={tp_size}: num_heads={config.num_heads}")

    print(
        f"\nBenchmarking SSM kernel "
        f"({config.num_mamba_layers} layers, {config.state_dtype})..."
    )
    print(
        f"Finding bandwidth saturation point "
        f"(threshold: {bw_threshold * 100:.1f}%)...\n"
    )

    batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
    results: list[BenchmarkResult] = []
    prev_bw, saturated = 0.0, False

    print(f"{'Batch':>8} | {'Time(ms)':>10} | {'BW(TB/s)':>10} | {'BW Δ':>10} | Status")
    print("-" * 65)

    for batch_size in batch_sizes:
        try:
            result = benchmark_ssm_kernel(config, batch_size, compute_dtype)
            results.append(result)

            bw_improvement = (
                (result.effective_bandwidth_tb_s - prev_bw) / prev_bw
                if prev_bw > 0
                else 1.0
            )

            if bw_improvement >= bw_threshold:
                status = "improving"
            elif not saturated:
                status, saturated = "SATURATED", True
            else:
                status = "plateau"

            print(
                f"{batch_size:>8} | {result.time_ms:>10.2f} | "
                f"{result.effective_bandwidth_tb_s:>10.2f} | "
                f"{bw_improvement:>+9.1%} | {status}"
            )

            prev_bw = result.effective_bandwidth_tb_s
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{batch_size:>8} | OOM")
            torch.cuda.empty_cache()
            break

    # Find where bandwidth reaches near-peak (more stable than threshold detection)
    max_bw = max(r.effective_bandwidth_tb_s for r in results)
    near_peak_threshold = near_peak_ratio * max_bw

    # Find smallest batch size that achieves near-peak bandwidth
    near_peak_batch = next(
        (
            r.batch_size
            for r in results
            if r.effective_bandwidth_tb_s >= near_peak_threshold
        ),
        results[-1].batch_size,
    )
    near_peak_batch *= tp_size

    # Apply headroom for overall throughput optimization
    # Round to nearest 64 for cleaner values
    recommended = int(near_peak_batch * throughput_headroom)
    recommended = ((recommended + 63) // 64) * 64

    return near_peak_batch, recommended, results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Mamba SSM kernel to find optimal max-num-seqs",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
    )
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", help="Batch sizes to test"
    )
    parser.add_argument(
        "--bw-threshold",
        type=float,
        default=0.005,
        help="Bandwidth improvement threshold for saturation detection",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for model tensors (default: bfloat16)",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    state_dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    config = get_config_from_model(
        args.model,
        state_dtype_map[args.mamba_ssm_cache_dtype],
        args.trust_remote_code,
    )
    compute_dtype = compute_dtype_map[args.dtype]

    print(
        f"GPU: {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
    )

    # Empirical headroom for overall throughput optimization (tested on B200)
    throughput_headroom = 2.0

    near_peak_batch, recommended, results = find_optimal_batch_size(
        config,
        args.batch_sizes,
        args.tensor_parallel_size,
        compute_dtype,
        args.bw_threshold,
        throughput_headroom,
    )

    max_bw = max(r.effective_bandwidth_tb_s for r in results)
    print(f"\nBandwidth plateau: {max_bw:.2f} TB/s (99% = {0.99 * max_bw:.2f} TB/s)")
    print(f"Near-peak batch size: {near_peak_batch}")

    print(f"\n{'=' * 50}")
    print(f"RECOMMENDED --max-num-seqs: {recommended}")
    print(f"{'=' * 50}")
    print(f"(near-peak batch {near_peak_batch} × {throughput_headroom:.1f}x headroom)")


if __name__ == "__main__":
    main()
