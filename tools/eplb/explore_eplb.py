#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple script to explore EPLB behavior on MoE models.

Captures expert load distribution and mappings before/after rearrangement.

Usage:
    python explore_eplb.py --model deepseek-ai/DeepSeek-V2-Lite --num-prompts 500
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import add_dataset_parser, get_samples
from vllm.config.parallel import EPLBConfig
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def has_mapping_changed(prev_stats, current_stats):
    """Detect if expert mapping has changed between two stats snapshots.

    Uses CUDA tensors for fast comparison of large mappings.
    """
    if not prev_stats or not current_stats:
        return False

    # Convert to CUDA tensors for fast comparison
    device = torch.device("cuda:0")

    # Compare physical_to_logical_map
    prev_map = torch.tensor(
        prev_stats.get("physical_to_logical_map", []), dtype=torch.int64, device=device
    )
    curr_map = torch.tensor(
        current_stats.get("physical_to_logical_map", []),
        dtype=torch.int64,
        device=device,
    )

    if not torch.equal(prev_map, curr_map):
        return True

    # Also compare logical_replica_count as it affects routing
    prev_replica = torch.tensor(
        prev_stats.get("logical_replica_count", []), dtype=torch.int64, device=device
    )
    curr_replica = torch.tensor(
        current_stats.get("logical_replica_count", []), dtype=torch.int64, device=device
    )
    return bool(not torch.equal(prev_replica, curr_replica))


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
            balancedness = mean_load / max_load if max_load > 0 else 0.0

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
        logger.exception("Error extracting stats: %s", e)
        return None


def generate_summary(stats, top_k_experts=10, tp_size=1):
    """Generate a summary with top-k expert load percentages and per-GPU loads.

    Returns a dict with:
    - model_summary: Total load across all layers with top-k experts and GPU loads
    - per_layer_summary: Per-layer summaries with top-k experts and GPU loads
    """
    if not stats:
        return None

    num_physical_experts = stats["num_physical_experts"]
    experts_per_gpu = num_physical_experts // tp_size

    # Aggregate logical load and physical load across all layers
    num_logical_experts = stats["num_logical_experts"]
    total_logical_load = [0.0] * num_logical_experts
    total_physical_load = [0.0] * num_physical_experts

    for layer_stat in stats["per_layer_stats"]:
        logical_load = layer_stat["logical_load"]
        physical_load = layer_stat["physical_load"]
        for i, load in enumerate(logical_load):
            total_logical_load[i] += load
        for i, load in enumerate(physical_load):
            total_physical_load[i] += load

    # Calculate per-GPU load (model-wide)
    gpu_loads = []
    total_load_all = sum(total_physical_load)
    gpu_load_values = []
    for gpu_id in range(tp_size):
        start_idx = gpu_id * experts_per_gpu
        end_idx = start_idx + experts_per_gpu
        gpu_load = sum(total_physical_load[start_idx:end_idx])
        gpu_percentage = (gpu_load / total_load_all * 100) if total_load_all > 0 else 0
        gpu_loads.append(gpu_percentage)
        gpu_load_values.append(gpu_load)

    # Calculate balancedness for GPU loads
    mean_gpu_load = (
        sum(gpu_load_values) / len(gpu_load_values) if gpu_load_values else 0
    )
    max_gpu_load = max(gpu_load_values) if gpu_load_values else 0
    gpu_balancedness = mean_gpu_load / max_gpu_load if max_gpu_load > 0 else 1.0

    model_summary = {
        "total_gpu_load": {
            "gpu_loads": gpu_loads,
            "balancedness": gpu_balancedness,
        },
    }

    # Per-layer summaries
    per_layer_summary = []
    for layer_stat in stats["per_layer_stats"]:
        layer_idx = layer_stat["layer"]
        logical_load = layer_stat["logical_load"]
        physical_load = layer_stat["physical_load"]
        total_load = sum(physical_load)

        # Calculate per-GPU load for this layer
        layer_gpu_loads = []
        layer_gpu_load_values = []
        for gpu_id in range(tp_size):
            start_idx = gpu_id * experts_per_gpu
            end_idx = start_idx + experts_per_gpu
            gpu_load = sum(physical_load[start_idx:end_idx])
            gpu_percentage = (gpu_load / total_load * 100) if total_load > 0 else 0
            layer_gpu_loads.append(gpu_percentage)
            layer_gpu_load_values.append(gpu_load)

        # Calculate balancedness for this layer's GPU loads
        mean_layer_gpu_load = (
            sum(layer_gpu_load_values) / len(layer_gpu_load_values)
            if layer_gpu_load_values
            else 0
        )
        max_layer_gpu_load = max(layer_gpu_load_values) if layer_gpu_load_values else 0
        layer_gpu_balancedness = (
            mean_layer_gpu_load / max_layer_gpu_load if max_layer_gpu_load > 0 else 1.0
        )

        # Calculate top-k experts for this layer
        expert_loads = [
            (expert_id, load) for expert_id, load in enumerate(logical_load) if load > 0
        ]
        expert_loads.sort(key=lambda x: x[1], reverse=True)

        top_k_list = []
        for expert_id, load in expert_loads[:top_k_experts]:
            percentage = (load / total_load * 100) if total_load > 0 else 0
            replica_count = stats["logical_replica_count"][layer_idx][expert_id]
            top_k_list.append(
                {
                    "expert_id": expert_id,
                    "percentage": percentage,
                    "replicas": replica_count,
                }
            )

        remaining_load = sum(load for _, load in expert_loads[top_k_experts:])
        remaining_pct = (remaining_load / total_load * 100) if total_load > 0 else 0

        per_layer_summary.append(
            {
                "layer": layer_idx,
                "gpu_loads": layer_gpu_loads,
                "gpu_balancedness": layer_gpu_balancedness,
                f"top_{top_k_experts}_experts": top_k_list,
                "remaining_experts": {
                    "count": len(expert_loads) - top_k_experts,
                    "percentage": remaining_pct,
                },
            }
        )

    return {
        "step": stats["current_step"],
        "step_interval": stats["step_interval"],
        "num_logical_experts": stats["num_logical_experts"],
        "num_physical_experts": stats["num_physical_experts"],
        "num_layers": stats["num_layers"],
        "model_summary": model_summary,
        "per_layer_summary": per_layer_summary,
    }


def analyze_and_save_rearrangement(
    before, after, tp_size, rearrangement_num, output_dir
):
    """Analyze expert movements and save per-layer details to files."""
    if not before or not after:
        logger.warning("Cannot analyze: missing stats")
        return None

    rearrange_dir = (
        Path(output_dir) / "rearrangements" / f"rearrangement_{rearrangement_num}"
    )
    rearrange_dir.mkdir(parents=True, exist_ok=True)

    all_layer_data = []
    total_experts_moved = 0
    total_gpu_movements = {}  # (from_gpu, to_gpu) -> count across all layers

    # Analyze movements for each layer
    for layer_idx in range(len(before["physical_to_logical_map"])):
        before_map = before["physical_to_logical_map"][layer_idx]
        after_map = after["physical_to_logical_map"][layer_idx]

        num_physical = len(before_map)
        experts_per_gpu = num_physical // tp_size

        # Track which logical expert is on which GPU before and after
        logical_gpu_map = {}

        for phys_idx in range(num_physical):
            gpu_id = phys_idx // experts_per_gpu
            before_logical = before_map[phys_idx]
            after_logical = after_map[phys_idx]

            # Track where this logical expert was before
            if before_logical not in logical_gpu_map:
                logical_gpu_map[before_logical] = {"before": set(), "after": set()}
            logical_gpu_map[before_logical]["before"].add(gpu_id)

            # Track where this logical expert is after
            if after_logical not in logical_gpu_map:
                logical_gpu_map[after_logical] = {"before": set(), "after": set()}
            logical_gpu_map[after_logical]["after"].add(gpu_id)

        # Count movements between GPUs
        movements_summary = {}  # (from_gpu, to_gpu) -> count
        moved_experts = []

        for logical_id, locations in logical_gpu_map.items():
            before_gpus = locations.get("before", set())
            after_gpus = locations.get("after", set())

            # Check for cross-GPU movements
            new_gpus = after_gpus - before_gpus  # GPUs where expert appeared
            removed_gpus = before_gpus - after_gpus  # GPUs where expert disappeared

            # Count as movement if expert was added to any new GPU(s)
            # This means weights had to be transferred to those GPU(s)
            if new_gpus:
                moved_experts.append(
                    {
                        "logical_id": logical_id,
                        "before_gpus": sorted(list(before_gpus)),
                        "after_gpus": sorted(list(after_gpus)),
                        "added_to": sorted(list(new_gpus)),
                        "removed_from": sorted(list(removed_gpus)),
                    }
                )

                # Track GPU-to-GPU movements
                # For each new GPU, we need to identify where the weights came from
                if removed_gpus:
                    # Expert was moved from some GPU(s) to new GPU(s)
                    for from_gpu in removed_gpus:
                        for to_gpu in new_gpus:
                            key = (from_gpu, to_gpu)
                            movements_summary[key] = movements_summary.get(key, 0) + 1
                            total_gpu_movements[key] = (
                                total_gpu_movements.get(key, 0) + 1
                            )
                else:
                    # Expert was replicated (added to new GPU but still on original GPU)
                    # Source could be any of the before_gpus
                    for source_gpu in before_gpus:
                        for to_gpu in new_gpus:
                            key = (source_gpu, to_gpu)
                            movements_summary[key] = movements_summary.get(key, 0) + 1
                            total_gpu_movements[key] = (
                                total_gpu_movements.get(key, 0) + 1
                            )

        total_experts_moved += len(moved_experts)

        # Save layer data
        layer_data = {
            "layer_idx": layer_idx,
            "num_experts_moved": len(moved_experts),
            "gpu_to_gpu_movements": {
                f"{k[0]}_to_{k[1]}": v for k, v in movements_summary.items()
            },
            "moved_experts": moved_experts,
        }
        all_layer_data.append(layer_data)

        # Save individual layer file
        layer_file = rearrange_dir / f"layer_{layer_idx}.json"
        with open(layer_file, "w") as f:
            json.dump(layer_data, f, indent=2)

    # Save summary file
    summary = {
        "rearrangement_num": rearrangement_num,
        "step_before": before["current_step"],
        "step_after": after["current_step"],
        "total_experts_moved": total_experts_moved,
        "total_gpu_movements": {
            f"{k[0]}_to_{k[1]}": v for k, v in total_gpu_movements.items()
        },
        "per_layer_summary": [
            {"layer": ld["layer_idx"], "experts_moved": ld["num_experts_moved"]}
            for ld in all_layer_data
        ],
    }

    summary_file = rearrange_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Saved rearrangement #%d details to %s", rearrangement_num, rearrange_dir
    )

    return summary


def print_rearrangement_summary(summary):
    """Print a summary of the rearrangement."""
    if not summary:
        return

    print(f"\n{'=' * 80}")
    print(
        f"Rearrangement #{summary['rearrangement_num']}: "
        f"Step {summary['step_before']} → Step {summary['step_after']}"
    )
    print(f"{'=' * 80}")

    print(f"\nTotal logical experts moved: {summary['total_experts_moved']}")

    if summary["total_gpu_movements"]:
        print("\nCross-GPU movements (aggregated across all layers):")
        for movement_key, count in sorted(summary["total_gpu_movements"].items()):
            from_gpu, to_gpu = movement_key.split("_to_")
            print(f"  GPU {from_gpu} → GPU {to_gpu}: {count} expert(s)")

    print("\nPer-layer summary:")
    for layer_summary in summary["per_layer_summary"]:
        if layer_summary["experts_moved"] > 0:
            print(
                f"Layer {layer_summary['layer']}: "
                f"{layer_summary['experts_moved']} expert(s) moved"
            )


def main(args):
    logger.info("=" * 80)
    logger.info("EPLB Exploration Script")
    logger.info("=" * 80)
    logger.info("Model: %s", args.model)
    logger.info("Number of prompts: %d", args.num_prompts)
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("TP Size: %d", args.tp)
    logger.info("EPLB Window Size: %d", args.eplb_window_size)
    logger.info("EPLB Step Interval: %d", args.eplb_step_interval)
    logger.info("Redundant Experts: %d", args.num_redundant_experts)
    logger.info("Seed: %d", args.seed)
    logger.info("=" * 80)

    # Load prompts using get_samples
    logger.info(
        "\nLoading %d prompts from %s dataset...", args.num_prompts, args.dataset_name
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Set backend if not set (for get_samples compatibility)
    if not hasattr(args, "backend"):
        args.backend = "vllm"

    sample_requests = get_samples(args, tokenizer)
    prompts = [req.prompt for req in sample_requests]

    logger.info("Loaded %d prompts", len(prompts))
    logger.info("Example prompt: %s...", prompts[0][:100])
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

    if args.dataset_name == "sonnet":
        args.output_len = args.sonnet_output_len
    elif args.dataset_name == "custom":
        args.output_len = args.custom_output_len
    elif args.dataset_name == "sharegpt":
        args.output_len = args.sharegpt_output_len
    elif args.dataset_name == "burstgpt":
        args.output_len = args.burstgpt_output_len
    elif args.dataset_name == "hf":
        args.output_len = args.hf_output_len
    elif args.dataset_name == "spec_bench":
        args.output_len = args.spec_bench_output_len
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.output_len,
    )

    # Get initial stats
    logger.info("\nGetting initial EPLB stats...")
    initial_stats = get_eplb_stats(llm)

    if not initial_stats:
        logger.error("EPLB is not enabled or stats unavailable!")
        return

    # Create output directories
    output_dir = Path(args.output_dir)
    detailed_dir = output_dir / "detailed_snapshots"
    summary_dir = output_dir / "summaries"
    detailed_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Track all snapshots
    snapshots = [initial_stats]
    previous_stats = initial_stats
    rearrangement_summaries = []

    # Save initial detailed snapshot and summary
    initial_detailed_file = detailed_dir / "snapshot_0_initial.json"
    with open(initial_detailed_file, "w") as f:
        json.dump(initial_stats, f, indent=2)

    initial_summary = generate_summary(
        initial_stats, top_k_experts=args.top_k_experts, tp_size=args.tp
    )
    initial_summary_file = summary_dir / "summary_0_initial.json"
    with open(initial_summary_file, "w") as f:
        json.dump(initial_summary, f, indent=2)

    logger.info("Saved initial detailed snapshot to %s", initial_detailed_file)
    logger.info("Saved initial summary to %s", initial_summary_file)

    logger.info("\nProcessing prompts and tracking rearrangements...")

    # Process prompts in small batches to track progress
    batch_size = args.batch_size
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        # Generate
        llm.generate(batch_prompts, sampling_params)

        # Check current step
        current_stats = get_eplb_stats(llm)
        if current_stats:
            current_step = current_stats["current_step"]
            logger.info(
                "Batch %d/%d: Step %d/%d",
                batch_idx + 1,
                num_batches,
                current_step,
                current_stats["step_interval"],
            )

            # Check if rearrangement just happened by comparing expert mappings
            if has_mapping_changed(previous_stats, current_stats):
                rearrangement_num = len(snapshots)
                logger.info("Rearrangement #%d detected!", rearrangement_num)
                snapshots.append(current_stats)

                # Save detailed snapshot
                detailed_file = (
                    detailed_dir / f"snapshot_{rearrangement_num}_after_rearrange.json"
                )
                with open(detailed_file, "w") as f:
                    json.dump(current_stats, f, indent=2)

                # Save summary
                current_summary = generate_summary(
                    current_stats, top_k_experts=args.top_k_experts, tp_size=args.tp
                )
                summary_file = (
                    summary_dir / f"summary_{rearrangement_num}_after_rearrange.json"
                )
                with open(summary_file, "w") as f:
                    json.dump(current_summary, f, indent=2)

                logger.info("Saved detailed snapshot to %s", detailed_file)
                logger.info("Saved summary to %s", summary_file)

                # Analyze and save rearrangement details
                rearrange_summary = analyze_and_save_rearrangement(
                    snapshots[-2], snapshots[-1], args.tp, rearrangement_num, output_dir
                )
                if rearrange_summary:
                    rearrangement_summaries.append(rearrange_summary)
                    print_rearrangement_summary(rearrange_summary)

            previous_stats = current_stats

    logger.info("\nGenerated %d outputs", len(prompts))

    # Final summary
    num_rearrangements = len(snapshots) - 1
    if num_rearrangements > 0:
        print(f"\n{'=' * 80}")
        print(f"FINAL SUMMARY: {num_rearrangements} rearrangement(s) occurred")
        print(f"{'=' * 80}")
        for i, summary in enumerate(rearrangement_summaries, 1):
            print(
                f"Rearrangement #{i}: {summary['total_experts_moved']} experts moved "
                f"(Step {summary['step_before']} → {summary['step_after']})"
            )
    else:
        logger.info("\nNo rearrangements occurred during this run.")
        logger.info(
            "Current step: %d/%d",
            snapshots[-1]["current_step"],
            snapshots[-1]["step_interval"],
        )
        logger.info("Try running with more prompts or lower step_interval")

    # Save overall summary
    overall_summary_file = output_dir / "overall_summary.json"
    overall_summary = {
        "num_rearrangements": num_rearrangements,
        "rearrangement_summaries": rearrangement_summaries,
    }
    with open(overall_summary_file, "w") as f:
        json.dump(overall_summary, f, indent=2)

    logger.info("\n%s", "=" * 80)
    logger.info("All results saved to %s", output_dir)
    logger.info("  - Detailed snapshots: %s", detailed_dir)
    logger.info("  - Summaries: %s", summary_dir)
    logger.info("  - Rearrangements: %s", output_dir / "rearrangements")
    logger.info("  - Overall summary: %s", overall_summary_file)
    logger.info("%s", "=" * 80)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Explore EPLB behavior on MoE models")

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Model name or path",
    )
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument(
        "--max-len", type=int, default=2048, help="Max model sequence length"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eplb_results",
        help="Output directory for all results",
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
    parser.add_argument(
        "--top-k-experts",
        type=int,
        default=10,
        help="Number of top loaded experts to display per layer",
    )

    # Add dataset-related arguments from benchmarks
    add_dataset_parser(parser)

    args = parser.parse_args()
    main(args)
