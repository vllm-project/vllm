#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple script to explore EPLB behavior on MoE models.

Captures expert load distribution and mappings before/after rearrangement.

Usage:
    python explore_eplb.py --model deepseek-ai/DeepSeek-V2-Lite --num-prompts 500
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import SonnetDataset
from vllm.config.parallel import EPLBConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def load_prompts(
    model_name: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    seed: int,
):
    dataset_path = Path(__file__).parent / "benchmarks" / "sonnet.txt"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Sonnet dataset not found at {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    assert tokenizer.chat_template or tokenizer.default_chat_template, (
        "Tokenizer/model must have chat template for sonnet dataset."
    )

    # Common kwargs for dataset instantiation
    common_kwargs = {
        "dataset_path": str(dataset_path),
        "random_seed": seed,
    }

    # Sample kwargs (same as in throughput.py for sonnet)
    sample_kwargs = {
        "tokenizer": tokenizer,
        "num_requests": num_prompts,
        "input_len": input_len,
        "output_len": output_len,
        "prefix_len": prefix_len,
        "return_prompt_formatted": True,
    }

    # Remove None values
    sample_kwargs = {k: v for k, v in sample_kwargs.items() if v is not None}

    # Instantiate and sample (same pattern as throughput.py)
    dataset = SonnetDataset(**common_kwargs)
    requests = dataset.sample(**sample_kwargs)

    return [req.prompt for req in requests]


def _get_eplb_stats_from_worker(worker):
    """Helper function to extract EPLB stats from worker (runs in worker process)."""
    model_runner = worker.model_runner

    if not hasattr(model_runner, "eplb_state") or model_runner.eplb_state is None:
        return None

    eplb_state = model_runner.eplb_state
    model = model_runner.model

    # Get current step info
    current_step = eplb_state.expert_rearrangement_step
    step_interval = eplb_state.expert_rearrangement_step_interval

    # Get expert load from window (move to CPU for transfer)
    expert_load_window = eplb_state.expert_load_window.cpu()
    total_load = expert_load_window.sum(dim=0)

    # Get mappings (move to CPU)
    physical_to_logical = eplb_state.physical_to_logical_map.cpu()
    logical_replica_count = eplb_state.logical_replica_count.cpu()

    # Return serializable data
    return {
        "current_step": current_step,
        "step_interval": step_interval,
        "physical_to_logical_map": physical_to_logical,
        "logical_replica_count": logical_replica_count,
        "expert_load_window": expert_load_window,
        "total_load": total_load,
        "num_layers": model.num_moe_layers,
        "num_logical_experts": model.num_logical_experts,
        "num_physical_experts": model.num_physical_experts,
    }


def get_eplb_stats(llm):
    """Extract key EPLB statistics via RPC to worker."""
    try:
        # Use collective_rpc to call the helper function on the driver worker
        results = llm.llm_engine.model_executor.collective_rpc(
            _get_eplb_stats_from_worker
        )

        # Get result from driver worker (first in list)
        worker_stats = results[0]

        if worker_stats is None:
            return None

        # Extract data from worker stats
        current_step = worker_stats["current_step"]
        step_interval = worker_stats["step_interval"]
        physical_to_logical = worker_stats["physical_to_logical_map"]
        logical_replica_count = worker_stats["logical_replica_count"]
        total_load = worker_stats["total_load"]

        # Compute balancedness per layer
        per_layer_stats = []
        for layer_idx in range(total_load.shape[0]):
            layer_load = total_load[layer_idx]

            # Map physical loads to logical experts
            logical_load = torch.zeros(
                logical_replica_count.shape[1], dtype=layer_load.dtype
            )
            for phys_idx, log_idx in enumerate(physical_to_logical[layer_idx]):
                logical_load[log_idx] += layer_load[phys_idx]

            # Calculate stats
            mean_load = layer_load.float().mean().item()
            max_load = layer_load.max().item()
            balancedness = mean_load / max_load if max_load > 0 else 1.0

            per_layer_stats.append(
                {
                    "layer": layer_idx,
                    "total_tokens": layer_load.sum().item(),
                    "mean_load": mean_load,
                    "max_load": max_load,
                    "min_load": layer_load.min().item(),
                    "balancedness": balancedness,
                    "physical_load": layer_load.tolist(),
                    "logical_load": logical_load.tolist(),
                }
            )

        return {
            "current_step": current_step,
            "step_interval": step_interval,
            "steps_until_rearrange": step_interval - current_step,
            "physical_to_logical_map": physical_to_logical.tolist(),
            "logical_replica_count": logical_replica_count.tolist(),
            "per_layer_stats": per_layer_stats,
            "num_layers": worker_stats["num_layers"],
            "num_logical_experts": worker_stats["num_logical_experts"],
            "num_physical_experts": worker_stats["num_physical_experts"],
        }

    except Exception as e:
        logger.error(f"Error extracting stats: {e}", exc_info=True)
        return None


def print_stats(stats, label="Stats"):
    """Print statistics in a readable format."""
    if not stats:
        print(f"\n{label}: No stats available")
        return

    print(f"\n{'=' * 80}")
    print(f"{label}")
    print(f"{'=' * 80}")
    print(
        f"Step: {stats['current_step']}/{stats['step_interval']} "
        f"(rearrange in {stats['steps_until_rearrange']} steps)"
    )
    print(
        f"Experts: {stats['num_logical_experts']} logical, "
        f"{stats['num_physical_experts']} physical"
    )
    print(f"Layers: {stats['num_layers']}")

    print(
        f"\n{'Layer':<8} {'Total Tokens':<15} {'Mean Load':<12} {'Max Load':<12} "
        f"{'Balancedness':<12}"
    )
    print("-" * 80)

    for layer_stat in stats["per_layer_stats"]:
        print(
            f"{layer_stat['layer']:<8} "
            f"{layer_stat['total_tokens']:<15.0f} "
            f"{layer_stat['mean_load']:<12.1f} "
            f"{layer_stat['max_load']:<12.0f} "
            f"{layer_stat['balancedness']:<12.3f}"
        )

    # Show expert mappings for first layer
    print("\nFirst Layer Expert Mappings (Physical → Logical):")
    phys_to_log = stats["physical_to_logical_map"][0]
    replica_counts = stats["logical_replica_count"][0]

    # Group by logical expert
    logical_to_physical = {}
    for phys_id, log_id in enumerate(phys_to_log):
        if log_id not in logical_to_physical:
            logical_to_physical[log_id] = []
        logical_to_physical[log_id].append(phys_id)

    print(f"{'Logical Expert':<18} {'Replicas':<10} {'Physical Expert IDs'}")
    print("-" * 60)
    for log_id in sorted(logical_to_physical.keys())[:10]:  # Show first 10
        phys_ids = logical_to_physical[log_id]
        print(f"Expert {log_id:<12} {len(phys_ids):<10} {phys_ids}")

    if len(logical_to_physical) > 10:
        print(f"... ({len(logical_to_physical) - 10} more)")


def compare_stats(before, after):
    """Compare before and after stats."""
    if not before or not after:
        print("\nCannot compare: missing stats")
        return

    print(f"\n{'=' * 80}")
    print("Comparison: BEFORE vs AFTER Rearrangement")
    print(f"{'=' * 80}")

    print(f"\n{'Layer':<8} {'Before Bal':<15} {'After Bal':<15} {'Improvement':<15}")
    print("-" * 80)

    improvements = []
    for i in range(len(before["per_layer_stats"])):
        before_bal = before["per_layer_stats"][i]["balancedness"]
        after_bal = after["per_layer_stats"][i]["balancedness"]
        improvement = (
            (after_bal - before_bal) / before_bal * 100 if before_bal > 0 else 0
        )
        improvements.append(improvement)

        print(f"{i:<8} {before_bal:<15.3f} {after_bal:<15.3f} {improvement:>+6.1f}%")

    print("-" * 80)
    avg_improvement = np.mean(improvements)
    print(f"{'Average':<8} {'':<15} {'':<15} {avg_improvement:>+6.1f}%")

    # Check mapping changes
    before_map = before["physical_to_logical_map"][0]
    after_map = after["physical_to_logical_map"][0]

    changes = sum(1 for b, a in zip(before_map, after_map) if b != a)
    print(
        f"\nMapping changes in first layer: {changes}/{len(before_map)} experts remapped"
    )


def main(args):
    # Set random seed (same as throughput.py)
    if args.seed is None:
        args.seed = 0
    random.seed(args.seed)

    logger.info("=" * 80)
    logger.info("EPLB Exploration Script")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Number of prompts: {args.num_prompts}")
    logger.info(f"TP Size: {args.tp}")
    logger.info(f"EPLB Window Size: {args.eplb_window_size}")
    logger.info(f"EPLB Step Interval: {args.eplb_step_interval}")
    logger.info(f"Redundant Experts: {args.num_redundant_experts}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 80)

    # Load prompts
    logger.info(f"\nLoading {args.num_prompts} prompts from sonnet dataset...")
    prompts = load_prompts(
        model_name=args.model,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        prefix_len=args.prefix_len,
        seed=args.seed,
    )
    logger.info(f"Loaded {len(prompts)} prompts")
    logger.info(f"Example prompt: {prompts[0][:100]}...")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    eplb_config = EPLBConfig(
        window_size=args.eplb_window_size,
        step_interval=args.eplb_step_interval,
        num_redundant_experts=args.num_redundant_experts,
        log_balancedness=True,
    )
    # Create LLM
    logger.info("\nInitializing LLM with EPLB...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enable_expert_parallel=True,
        enable_eplb=True,
        trust_remote_code=True,
        max_model_len=args.max_len,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        eplb_config=eplb_config,
    )

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.output_len,
    )

    # Get initial stats
    logger.info("\nGetting initial EPLB stats...")
    stats_before = get_eplb_stats(llm)

    if not stats_before:
        logger.error("EPLB is not enabled or stats unavailable!")
        return

    print_stats(stats_before, "BEFORE First Rearrangement")

    # Calculate how many prompts to process before rearrangement
    steps_until_rearrange = stats_before["steps_until_rearrange"]
    logger.info(
        f"\nProcessing prompts until rearrangement ({steps_until_rearrange} steps)..."
    )

    # Process prompts in small batches to track progress
    batch_size = args.batch_size
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    stats_after_rearrange = None
    rearrangement_happened = False

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        # Generate
        outputs = llm.generate(batch_prompts, sampling_params)

        # Check current step
        current_stats = get_eplb_stats(llm)
        if current_stats:
            current_step = current_stats["current_step"]
            logger.info(
                f"Batch {batch_idx + 1}/{num_batches}: Step {current_step}/{current_stats['step_interval']}"
            )

            # Check if rearrangement just happened (step reset to 0 or very small)
            if (
                not rearrangement_happened
                and current_step < stats_before["current_step"]
            ):
                logger.info("\n🔄 Rearrangement detected!")
                stats_after_rearrange = current_stats
                rearrangement_happened = True

    logger.info(f"\nGenerated {len(prompts)} outputs")

    # Get final stats
    logger.info("\nGetting final EPLB stats...")
    stats_final = get_eplb_stats(llm)

    if rearrangement_happened and stats_after_rearrange:
        print_stats(stats_after_rearrange, "AFTER Rearrangement")
        compare_stats(stats_before, stats_after_rearrange)
    else:
        logger.info("\nRearrangement did not occur during this run.")
        logger.info(
            f"Current step: {stats_final['current_step']}/{stats_final['step_interval']}"
        )
        logger.info("Try running with more prompts or lower step_interval")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "before": stats_before,
            "after": stats_after_rearrange if rearrangement_happened else None,
            "final": stats_final,
            "rearrangement_occurred": rearrangement_happened,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore EPLB behavior on MoE models")

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Model name or path",
    )
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument(
        "--num-prompts", type=int, default=500, help="Number of prompts to process"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=550,
        help="Input length for sonnet dataset (default: 550)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=150,
        help="Output length for sonnet dataset (default: 150)",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Prefix length for sonnet dataset (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for dataset sampling (default: 0)",
    )
    parser.add_argument(
        "--max-len", type=int, default=2048, help="Max model sequence length"
    )
    parser.add_argument(
        "--output", type=str, default="eplb_results.json", help="Output JSON file path"
    )
    parser.add_argument(
        "--eplb-window-size",
        type=int,
        default=100,
        help="EPLB window size for load tracking",
    )
    parser.add_argument(
        "--eplb-step-interval",
        type=int,
        default=200,
        help="EPLB rearrangement interval (steps)",
    )
    parser.add_argument(
        "--num-redundant-experts",
        type=int,
        default=2,
        help="Number of redundant experts for EPLB",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for processing prompts"
    )

    args = parser.parse_args()
    main(args)
