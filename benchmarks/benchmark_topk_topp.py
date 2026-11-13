#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark top-k/top-p filtering with Triton, FlashInfer, and optional PyTorch.

All implementations return masked logits. Timings include a fresh-input copy.
FlashInfer timings include softmax and mask reconstruction when its APIs require them.
L2 is flushed before each measured iteration, outside the timed region.

Scenarios:
- top_k only (whole batch, partial batch)
- top_p only (whole batch, partial batch)
- mix of top_k and top_p
- optional sparse inputs via --num-finite-logits
"""

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import flashinfer.sampling as fi_sampling
import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import (
    apply_top_k_top_p_triton,
    reset_buffer_cache,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    batch_size: int
    vocab_size: int
    # k and p can be tensors or None
    k_values: torch.Tensor | None  # [batch_size] or None
    p_values: torch.Tensor | None  # [batch_size] or None
    description: str
    ops_pct: float = 0.0  # Percentage of ops relative to batch size


def calculate_ops_pct(
    k_values: torch.Tensor | None,
    p_values: torch.Tensor | None,
    vocab_size: int,
    batch_size: int,
) -> float:
    """Calculate the percentage of active top-k and top-p operations.

    Returns percentage where 100% = batch_size ops.
    E.g., if all rows have both top-k and top-p active, returns 200%.
    """
    active_ops = 0

    if k_values is not None:
        # Count rows where k < vocab_size (active top-k filtering)
        active_ops += (k_values < vocab_size).sum().item()

    if p_values is not None:
        # Count rows where p < 1.0 (active top-p filtering)
        active_ops += (p_values < 1.0).sum().item()

    return (active_ops / batch_size) * 100 if batch_size > 0 else 0.0


def create_logits(
    batch_size: int,
    vocab_size: int,
    device: str = "cuda",
    num_finite_logits: int | None = None,
) -> torch.Tensor:
    """Create random logits mimicking a realistic LLM distribution.

    Uses a Zipf-like probability distribution (rank^-1.1) converted to logits
    via log, then randomly permuted per row. This produces a peaked distribution
    where a small number of tokens capture most probability mass, similar to
    real model outputs. When num_finite_logits is set, retain that many ranks
    before permuting their positions independently in each row.
    """
    # Create Zipf-like probabilities: p(rank) ~ rank^(-alpha)
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32, device=device)
    probs = ranks.pow(-1.1)
    probs = probs / probs.sum()

    # Convert to logits (log-probabilities, unnormalized is fine)
    base_logits = probs.log()
    if num_finite_logits is not None:
        base_logits[num_finite_logits:] = -float("inf")

    # Broadcast to batch and randomly permute each row
    logits = base_logits.unsqueeze(0).expand(batch_size, -1).clone()
    for i in range(batch_size):
        logits[i] = logits[i, torch.randperm(vocab_size, device=device)]

    return logits


def apply_top_k_top_p_flashinfer_mask(logits, k, p):
    """Keep top-k in logits; adapt probability-based top-p back to logits."""
    if k is not None:
        logits = fi_sampling.top_k_mask_logits(logits, k)
    if p is not None:
        # Top-p renormalization requires softmax, even after logits-based top-k.
        probs = logits.softmax(dim=-1)
        probs = fi_sampling.top_p_renorm_probs(probs, p, is_deterministic=True)
        # Return the selected original logits, not normalized probabilities.
        logits = logits.masked_fill(probs == 0, -float("inf"))
    return logits


IMPLEMENTATIONS = {
    "triton": apply_top_k_top_p_triton,
    "flashinfer-mask": apply_top_k_top_p_flashinfer_mask,
    "pytorch": apply_top_k_top_p_pytorch,
}


def measure_memory() -> tuple[int, int]:
    """Return (allocated, reserved) memory in bytes."""
    torch.accelerator.synchronize()
    return (
        torch.accelerator.memory_allocated(),
        torch.accelerator.max_memory_allocated(),
    )


def reset_memory_stats():
    """Reset peak memory statistics."""
    reset_buffer_cache()
    torch.accelerator.reset_peak_memory_stats()
    torch.accelerator.empty_cache()
    gc.collect()


def create_benchmark_configs(
    batch_sizes: list[int],
    vocab_sizes: list[int],
    device: str = "cuda",
) -> list[BenchmarkConfig]:
    """Create all benchmark configurations."""
    configs = []

    for vocab_size in vocab_sizes:
        for batch_size in batch_sizes:
            # 1. Top-k only - whole batch (all rows have k < vocab_size)
            k_all = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
            configs.append(
                BenchmarkConfig(
                    name=f"topk_whole_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_all,
                    p_values=None,
                    description=f"Top-k only (whole batch, k=50), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_all, None, vocab_size, batch_size),
                )
            )

            # 2. Top-k only - partial batch (half have k=50, half have k=vocab_size)
            k_partial = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
            k_partial[batch_size // 2 :] = vocab_size  # No filtering for second half
            configs.append(
                BenchmarkConfig(
                    name=f"topk_partial_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_partial,
                    p_values=None,
                    description=f"Top-k only (partial batch, 50% k=50, 50% k=vocab), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_partial, None, vocab_size, batch_size),
                )
            )

            # 3. Top-p only - whole batch (all rows have p < 1.0)
            p_all = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
            configs.append(
                BenchmarkConfig(
                    name=f"topp_whole_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=None,
                    p_values=p_all,
                    description=f"Top-p only (whole batch, p=0.9), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(None, p_all, vocab_size, batch_size),
                )
            )

            # 4. Top-p only - partial batch (half have p=0.9, half have p=1.0)
            p_partial = torch.full(
                (batch_size,), 0.9, dtype=torch.float32, device=device
            )
            p_partial[batch_size // 2 :] = 1.0  # No filtering for second half
            configs.append(
                BenchmarkConfig(
                    name=f"topp_partial_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=None,
                    p_values=p_partial,
                    description=f"Top-p only (partial batch, 50% p=0.9, 50% p=1.0), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(None, p_partial, vocab_size, batch_size),
                )
            )

            # 5. Mix of top-k and top-p (both applied to whole batch)
            k_mix = torch.full((batch_size,), 100, dtype=torch.int32, device=device)
            p_mix = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
            configs.append(
                BenchmarkConfig(
                    name=f"topk_topp_whole_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_mix,
                    p_values=p_mix,
                    description=f"Top-k + Top-p (whole batch, k=100, p=0.9), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_mix, p_mix, vocab_size, batch_size),
                )
            )

            # 6. Mix with partial application (some rows k only, some p only, some both)
            k_mixed = torch.full(
                (batch_size,), vocab_size, dtype=torch.int32, device=device
            )
            p_mixed = torch.full((batch_size,), 1.0, dtype=torch.float32, device=device)
            # First third: k only
            third = batch_size // 3
            k_mixed[:third] = 50
            # Second third: p only
            p_mixed[third : 2 * third] = 0.5
            # Last third: both k and p
            k_mixed[2 * third :] = 100
            p_mixed[2 * third :] = 0.9
            configs.append(
                BenchmarkConfig(
                    name=f"mixed_partial_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_mixed,
                    p_values=p_mixed,
                    description=f"Mixed partial (1/3 k=50, 1/3 p=0.9, 1/3 both), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_mixed, p_mixed, vocab_size, batch_size),
                )
            )

    return configs


def format_memory(bytes_val: int) -> str:
    """Format memory in human-readable form."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / (1024**3):.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / (1024**2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} B"


def run_benchmark(
    configs: list[BenchmarkConfig],
    implementations: list[str],
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
    verbose: bool = True,
    num_finite_logits: int | None = None,
    check: bool = False,
    output: Path | None = None,
):
    """Run selected implementations with identical inputs and masked-logit outputs."""
    results = []

    for index, config in enumerate(configs):
        logits = create_logits(
            config.batch_size,
            config.vocab_size,
            num_finite_logits=num_finite_logits,
        )
        checks = {}
        if check:
            checks = check_implementations(logits, config, implementations)
        k, p = config.k_values, config.p_values
        timings = {}
        for name in implementations:
            func = IMPLEMENTATIONS[name]
            reset_memory_stats()
            times = bench_gpu_time_with_cupti(
                # note: measured runtime includes logits.clone()
                lambda func=func, logits=logits, k=k, p=p: func(logits.clone(), k, p),
                dry_run_iters=warmup_iters,
                repeat_iters=benchmark_iters,
                use_cuda_graph=False,  # sampling is outside cuda graph
                cold_l2_cache=True,
            )
            elapsed = sum(times) / len(times)
            _, memory = measure_memory()
            timings[name] = dict(time_ms=elapsed, peak_memory=memory)
        row = create_result_row(config, timings, checks)
        results.append(row)
        if output is not None:
            with output.open("a") as file:
                file.write(json.dumps(row) + "\n")
        if verbose:
            print(f"Completed {index + 1}/{len(configs)}: {config.name}", flush=True)
        del logits
        reset_memory_stats()

    return results


def check_implementations(
    logits: torch.Tensor,
    config: BenchmarkConfig,
    implementations: list[str],
) -> dict:
    """Compare filtered distributions against PyTorch outside the timed region."""
    k, p = config.k_values, config.p_values
    reference = apply_top_k_top_p_pytorch(logits.clone(), k, p).softmax(dim=-1)
    checks = {}
    for name in implementations:
        actual = IMPLEMENTATIONS[name](logits.clone(), k, p).softmax(dim=-1)
        total_variation = (actual - reference).abs().sum(-1).max().item() / 2
        checks[name] = {
            "close": torch.allclose(actual, reference, rtol=1e-5, atol=1e-6),
            "max_total_variation": total_variation,
        }
        if not checks[name]["close"]:
            print(f"Distribution difference: {name} {config.name} {checks[name]}")
        del actual
    return checks


def print_summary_table(results: list[dict]):
    """Print the same result rows saved to the output file."""
    print()
    print(pd.DataFrame(results).to_string(index=False, float_format="{:.3g}".format))


def create_result_row(config, timings, checks) -> dict:
    """Build a shared row for the summary table and JSONL output."""
    row = {
        "Scenario": config.name.split("_b")[0],
        "Batch": config.batch_size,
        "Vocab": config.vocab_size,
        "Ops%": f"{config.ops_pct:.0f}%",
    }
    row.update({f"{name} (ms)": data["time_ms"] for name, data in timings.items()})
    for name, data in timings.items():
        row[f"{name} mem"] = format_memory(data["peak_memory"])
    for name, data in checks.items():
        row[f"{name} check"] = data["close"]
        row[f"{name} max TV"] = data["max_total_variation"]
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton and FlashInfer top-k/top-p filtering"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 128, 512, 1024, 2048],
        help="Batch sizes to test (default: 1 4 16 64)",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[32768, 131072],  # 32k, 128k
        help="Vocabulary sizes to test (default: 32768 131072)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=20,
        help="Number of benchmark iterations (default: 20)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary table",
    )

    parser.add_argument(
        "--num-finite-logits",
        type=int,
        help="Finite logits per row; remaining entries are -inf (default: dense)",
    )

    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=list(IMPLEMENTATIONS),
        default=["triton", "flashinfer-mask"],
        help="Implementations to compare (PyTorch is an optional reference)",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check filtered distributions against PyTorch outside the timed region",
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[
            "topk_whole",
            "topk_partial",
            "topp_whole",
            "topp_partial",
            "topk_topp_whole",
            "mixed_partial",
        ],
        help="Run only the selected scenarios (default: all)",
    )
    parser.add_argument("--output", type=Path, help="Save incremental JSONL results")

    args = parser.parse_args()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.open("x").close()
    if args.num_finite_logits is not None and not (
        0 <= args.num_finite_logits <= min(args.vocab_sizes)
    ):
        parser.error("--num-finite-logits must be between 0 and each vocabulary size")
    torch.manual_seed(0)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        return

    # Create configs
    configs = create_benchmark_configs(args.batch_sizes, args.vocab_sizes)
    if args.scenarios is not None:
        configs = [c for c in configs if c.name.split("_b")[0] in args.scenarios]

    # Run benchmarks
    results = run_benchmark(
        configs,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        verbose=not args.quiet,
        num_finite_logits=args.num_finite_logits,
        implementations=args.implementations,
        check=args.check,
        output=args.output,
    )

    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
