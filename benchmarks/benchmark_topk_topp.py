#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark top-k/top-p filtering with Triton, FlashInfer, and optional PyTorch.

All implementations return masked logits. Timings include a fresh-input copy.
FlashInfer timings include candidate filtering and dense masked-logit reconstruction.
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

import flashinfer
import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.triton_utils import tl, triton
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
    distribution: str = "zipf",
) -> torch.Tensor:
    """Create random logits mimicking a realistic LLM distribution.

    Uses a Zipf-like probability distribution (rank^-1.1) converted to logits
    via log, then randomly permuted per row. This produces a peaked distribution
    where a small number of tokens capture most probability mass, similar to
    real model outputs. When num_finite_logits is set, retain that many ranks
    before permuting their positions independently in each row.
    """
    if distribution != "zipf":
        logits = torch.randn(batch_size, vocab_size, device=device)
        if distribution == "bf16-rounded":
            logits = logits.bfloat16().float()
        if num_finite_logits is None:
            return logits
        logits[:, num_finite_logits:] = -float("inf")
        for i in range(batch_size):
            logits[i] = logits[i, torch.randperm(vocab_size, device=device)]
        return logits

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


def apply_top_k_top_p_flashinfer_fastpath(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    *,
    top_k_min: int,
    top_k_max: int,
) -> torch.Tensor:
    """Filter device tensors with required host bounds on per-row k.

    Every entry of k must lie in [top_k_min, top_k_max].
    """
    if can_use_fi_topk_topp_fastpath(logits, k, top_k_min, top_k_max):
        return apply_fi_topk_topp_fastpath(
            logits, k, p, top_k_min=top_k_min, top_k_max=top_k_max
        )
    raise NotImplementedError()


def can_use_fi_topk_topp_fastpath(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    top_k_min: int,
    top_k_max: int,
) -> bool:
    # Match FlashInfer's top-k-first gate, with no indices mapping.
    return (
        k is not None
        and k.shape == (logits.shape[0],)
        and 0 < top_k_min <= top_k_max <= 256
        and logits.shape[-1] >= 65536
        and top_k_max < logits.shape[-1]
    )


def apply_fi_topk_topp_fastpath(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    *,
    top_k_min: int,
    top_k_max: int,
) -> torch.Tensor:
    assert k is not None
    values, indices = flashinfer.top_k(
        logits.contiguous(),
        top_k_max,
        sorted=p is not None or top_k_min != top_k_max,
        deterministic=True,
        tie_break=1,
        dsa_graph_safe=logits.shape[-1] % 4 != 0,
    )
    output = torch.full_like(logits, -float("inf"))
    if p is None and top_k_min == top_k_max:
        return output.scatter_(1, indices, values)
    _fi_candidates_to_logits_kernel[(logits.shape[0],)](
        values,
        indices,
        k,
        p,
        output,
        values.stride(0),
        values.stride(1),
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        k.stride(0),
        p.stride(0) if p is not None else 0,
        top_k_max,
        p is not None,
        top_k_min == top_k_max,
        triton.next_power_of_2(top_k_max),
        num_warps=4,
    )
    return output


@triton.jit
def _fi_candidates_to_logits_kernel(
    values_ptr,
    indices_ptr,
    k_ptr,
    p_ptr,
    output_ptr,
    value_row_stride,
    value_col_stride,
    index_row_stride,
    index_col_stride,
    output_row_stride,
    output_col_stride,
    k_stride,
    p_stride,
    TOP_K_MAX: tl.constexpr,
    HAS_P: tl.constexpr,
    UNIFORM_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Filter [batch, top_k_max] FP32 candidates and scatter retained logits."""
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    rank = TOP_K_MAX - 1 - offset
    valid = offset < TOP_K_MAX
    value = tl.load(
        values_ptr + row * value_row_stride + rank * value_col_stride,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    index = tl.load(
        indices_ptr + row * index_row_stride + rank * index_col_stride,
        mask=valid,
        other=0,
    )
    if not UNIFORM_K:
        row_k = tl.load(k_ptr + row * k_stride)
        value = tl.where(rank < row_k, value, -float("inf"))
    keep = valid
    if HAS_P:
        weight = tl.exp(value - tl.max(value, 0))
        probability = weight / tl.sum(weight, 0)
        cumulative = tl.cumsum(probability, 0)
        p = tl.load(p_ptr + row * p_stride)
        keep = valid & (~(cumulative <= 1.0 - p) | (offset == TOP_K_MAX - 1))
    tl.store(
        output_ptr + row * output_row_stride + index * output_col_stride,
        value,
        mask=keep,
    )


IMPLEMENTATIONS = {
    "triton": apply_top_k_top_p_triton,
    "flashinfer-fastpath": apply_top_k_top_p_flashinfer_fastpath,
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
    top_k: int | None = None,
    top_p: float | None = None,
) -> list[BenchmarkConfig]:
    """Create all benchmark configurations."""
    configs = []
    k_only = 50 if top_k is None else top_k
    k_both = 100 if top_k is None else top_k
    p_value = 0.9 if top_p is None else top_p
    p_mixed_value = 0.5 if top_p is None else top_p

    for vocab_size in vocab_sizes:
        for batch_size in batch_sizes:
            # 1. Top-k only - whole batch (all rows have k < vocab_size)
            k_all = torch.full((batch_size,), k_only, dtype=torch.int32, device=device)
            configs.append(
                BenchmarkConfig(
                    name=f"topk_whole_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_all,
                    p_values=None,
                    description=f"Top-k only (whole batch, k={k_only}), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_all, None, vocab_size, batch_size),
                )
            )

            # 2. Top-k only - partial batch (half have k=50, half have k=vocab_size)
            k_partial = torch.full(
                (batch_size,), k_only, dtype=torch.int32, device=device
            )
            k_partial[batch_size // 2 :] = vocab_size  # No filtering for second half
            configs.append(
                BenchmarkConfig(
                    name=f"topk_partial_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_partial,
                    p_values=None,
                    description=f"Top-k only (partial batch, 50% k={k_only}, "
                    f"50% k=vocab), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_partial, None, vocab_size, batch_size),
                )
            )

            # 3. Top-p only - whole batch (all rows have p < 1.0)
            p_all = torch.full(
                (batch_size,), p_value, dtype=torch.float32, device=device
            )
            configs.append(
                BenchmarkConfig(
                    name=f"topp_whole_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=None,
                    p_values=p_all,
                    description=f"Top-p only (whole batch, p={p_value}), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(None, p_all, vocab_size, batch_size),
                )
            )

            # 4. Top-p only - partial batch (half have p=0.9, half have p=1.0)
            p_partial = torch.full(
                (batch_size,), p_value, dtype=torch.float32, device=device
            )
            p_partial[batch_size // 2 :] = 1.0  # No filtering for second half
            configs.append(
                BenchmarkConfig(
                    name=f"topp_partial_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=None,
                    p_values=p_partial,
                    description=f"Top-p only (partial batch, 50% p={p_value}, "
                    f"50% p=1.0), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(None, p_partial, vocab_size, batch_size),
                )
            )

            # 5. Mix of top-k and top-p (both applied to whole batch)
            k_mix = torch.full((batch_size,), k_both, dtype=torch.int32, device=device)
            p_mix = torch.full(
                (batch_size,), p_value, dtype=torch.float32, device=device
            )
            configs.append(
                BenchmarkConfig(
                    name=f"topk_topp_whole_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_mix,
                    p_values=p_mix,
                    description=f"Top-k + Top-p (whole batch, k={k_both}, "
                    f"p={p_value}), "
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
            k_mixed[:third] = k_only
            # Second third: p only
            p_mixed[third : 2 * third] = p_mixed_value
            # Last third: both k and p
            k_mixed[2 * third :] = k_both
            p_mixed[2 * third :] = p_value
            configs.append(
                BenchmarkConfig(
                    name=f"mixed_partial_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_mixed,
                    p_values=p_mixed,
                    description=f"Mixed partial (1/3 k={k_only}, "
                    f"1/3 p={p_mixed_value}, 1/3 k={k_both}/p={p_value}), "
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
    distribution: str = "zipf",
):
    """Run selected implementations with identical inputs and masked-logit outputs."""
    results = []

    for index, config in enumerate(configs):
        logits = create_logits(
            config.batch_size,
            config.vocab_size,
            num_finite_logits=num_finite_logits,
            distribution=distribution,
        )
        checks = {}
        if check:
            checks = check_implementations(logits, config, implementations)
        k, p = config.k_values, config.p_values
        timings = {}
        for name in implementations:
            func = IMPLEMENTATIONS[name]
            kwargs = prepare_kwargs(name, k, p, config.vocab_size)
            reset_memory_stats()
            times = bench_gpu_time_with_cupti(
                # note: measured runtime includes logits.clone()
                lambda func=func, logits=logits, k=k, p=p, kwargs=kwargs: func(
                    logits.clone(), k, p, **kwargs
                ),
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


def prepare_kwargs(
    name: str, k: torch.Tensor | None, p: torch.Tensor | None, vocab_size: int
) -> dict[str, bool | int]:
    """Resolve host dispatch hints outside timing, preserving device k/p inputs."""
    if name == "flashinfer-fastpath":
        if k is None:
            return {"top_k_min": 0, "top_k_max": 0}
        values = k.tolist()
        if not values or any(value <= 0 for value in values):
            return {"top_k_min": 0, "top_k_max": 0}
        return {"top_k_min": min(values), "top_k_max": max(values)}
    if name != "triton":
        return {}
    has_topp_only_rows = False
    if p is not None:
        topp_only = p < 1
        if k is not None:
            topp_only &= k >= vocab_size
        has_topp_only_rows = bool(topp_only.any().item())
    return {"has_topp_only_rows": has_topp_only_rows}


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
        kwargs = prepare_kwargs(name, k, p, config.vocab_size)
        actual = IMPLEMENTATIONS[name](logits.clone(), k, p, **kwargs).softmax(dim=-1)
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
    parser.add_argument("--top-k", type=int, help="Override active top-k values")
    parser.add_argument("--top-p", type=float, help="Override active top-p values")
    parser.add_argument(
        "--distribution",
        choices=["zipf", "normal", "bf16-rounded"],
        default="zipf",
        help="FP32 logit distribution (default: zipf)",
    )

    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=list(IMPLEMENTATIONS),
        default=["triton", "flashinfer-fastpath"],
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
    if args.top_k is not None and not 0 < args.top_k <= min(args.vocab_sizes):
        parser.error("--top-k must be between 1 and each vocabulary size")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
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
    configs = create_benchmark_configs(
        args.batch_sizes, args.vocab_sizes, top_k=args.top_k, top_p=args.top_p
    )
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
        distribution=args.distribution,
    )

    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
