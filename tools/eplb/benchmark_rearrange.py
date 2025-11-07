#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark EPLB expert weights rearrangement.

Example launch (torchrun):

    torchrun \
      --nproc_per_node=4 \
      /home/ilmarkov/repos/vllm/tools/eplb/benchmark_rearrange.py \
      --model deepseek-ai/DeepSeek-V2-Lite \
      --new-state-file /path/to/eplb_state.json

The provided state file must be compatible with vLLM's EPLB state format
saved via EplbState.save_to_file (contains physical_to_logical_map and metadata).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from vllm.distributed.eplb.rebalance_execute import (
    rearrange_expert_weights_inplace,
)
from vllm.transformers_utils.config import get_config, get_hf_text_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EPLB rearrange_expert_weights_inplace",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name or path. If provided, num-layers, num-logical-experts, "
            "and hidden-sizes will be inferred from its config unless overridden."
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override number of MoE layers (otherwise inferred from model)",
    )
    parser.add_argument(
        "--num-redundant-experts",
        type=int,
        default=0,
        help="Number of redundant experts added globally",
    )
    parser.add_argument(
        "--new-state-file",
        type=str,
        required=True,
        help=(
            "Path to EPLB state JSON file containing "
            "physical_to_logical_map and metadata"
        ),
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10,
        help="Number of rearrangement timing iterations",
    )
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_initial_global_physical_to_logical_map(
    num_logical_experts: int,
    num_redundant_experts: int,
) -> list[int]:
    mapping = list(range(num_logical_experts))
    mapping += [i % num_logical_experts for i in range(num_redundant_experts)]
    return mapping


def _get_first_present(cfg: object, names: list[str]) -> int | None:
    for name in names:
        if hasattr(cfg, name):
            val = getattr(cfg, name)
            if isinstance(val, int) and val > 0:
                return val
    return None


def infer_params_from_model(
    model: str,
) -> tuple[int, int, list[int]]:
    """
    Returns (num_layers, num_logical_experts, hidden_sizes)
    inferred from the model's transformers config.
    """
    cfg = get_config(
        model=model,
        trust_remote_code=True,
        revision=None,
        config_format="auto",
    )
    text_cfg = get_hf_text_config(cfg)

    num_layers = _get_first_present(
        text_cfg,
        [
            "num_hidden_layers",
            "n_layer",
            "num_layers",
        ],
    )
    if num_layers is None:
        raise ValueError("Could not infer num_layers from model config")

    # Logical experts (router experts)
    num_logical = _get_first_present(
        text_cfg,
        [
            "n_routed_experts",
            "num_experts",
            "n_experts",
            "moe_num_experts",
            "num_local_experts",
            "router_num_experts",
        ],
    )
    if num_logical is None:
        raise ValueError(
            "Could not infer number of experts from model config. "
            "Please pass --num-logical-experts explicitly."
        )

    # Use hidden_size if available to shape synthetic weights (twice for up/down)
    hidden = _get_first_present(
        text_cfg,
        [
            "hidden_size",
            "n_embd",
            "d_model",
        ],
    )
    if hidden is None:
        hidden = 65536
    hidden_sizes = [hidden, hidden]

    return int(num_layers), int(num_logical), hidden_sizes


# def load_state_from_file(
#     path: str,
#     device: torch.device,
# ) -> tuple[torch.Tensor, int, int, int]:
#     """Load EPLB state from JSON once per rank; no broadcast.

#     Returns:
#         (new_map, saved_layers, saved_logical, saved_physical)
#     """
#     with open(path) as f:
#         state = json.load(f)
#     saved_layers = int(state["num_moe_layers"])
#     saved_logical = int(state["num_logical_experts"])
#     saved_physical = int(state["num_physical_experts"])
#     phy2log_list = state["physical_to_logical_map"]
#     new_map = torch.tensor(phy2log_list, dtype=torch.int64, device=device)
#     return new_map, saved_layers, saved_logical, saved_physical


def load_state_broadcast(
    path: str,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int]:
    """Load EPLB state on rank 0, broadcast meta and map to all ranks (CUDA).

    Returns:
        (new_map, saved_layers, saved_logical, saved_physical)
    """
    rank = dist.get_rank()

    # Broadcast metadata first: [layers, logical, physical]
    meta = torch.empty(3, dtype=torch.int64, device=device)
    if rank == 0:
        with open(path) as f:
            state = json.load(f)
        saved_layers = int(state["num_moe_layers"])
        saved_logical = int(state["num_logical_experts"])
        saved_physical = int(state["num_physical_experts"])
        meta[0] = saved_layers
        meta[1] = saved_logical
        meta[2] = saved_physical
    dist.broadcast(meta, src=0)

    saved_layers = int(meta[0].item())
    saved_logical = int(meta[1].item())
    saved_physical = int(meta[2].item())

    # Allocate map and broadcast it
    new_map = torch.empty(
        (saved_layers, saved_physical), dtype=torch.int64, device=device
    )
    if rank == 0:
        phy2log_list = state["physical_to_logical_map"]
        new_map.copy_(torch.tensor(phy2log_list, dtype=torch.int64, device=device))
    dist.broadcast(new_map, src=0)

    return new_map, saved_layers, saved_logical, saved_physical


def create_expert_weights(
    num_layers: int,
    num_local_experts: int,
    hidden_sizes: list[int],
    rank: int,
    global_mapping: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    expert_weights: list[list[torch.Tensor]] = []
    for layer in range(num_layers):
        layer_weights: list[torch.Tensor] = []
        for weight_idx, hidden_size in enumerate(hidden_sizes):
            weight_tensor = torch.empty(
                num_local_experts, hidden_size, device=device, dtype=dtype
            )
            for local_expert in range(num_local_experts):
                global_pos = rank * num_local_experts + local_expert
                logical_id = int(global_mapping[layer, global_pos].item())
                base_value = logical_id * 1000 + layer * 100 + weight_idx * 10
                weight_tensor[local_expert].copy_(
                    torch.arange(
                        base_value,
                        base_value + hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                )
            layer_weights.append(weight_tensor)
        expert_weights.append(layer_weights)
    return expert_weights


def main() -> None:
    args = parse_args()
    rank, world_size, _ = init_distributed()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    new_map, saved_layers, saved_logical, saved_physical = load_state_broadcast(
        args.new_state_file, device
    )

    inferred_layers = inferred_logical = None
    inferred_hidden_sizes = [8192, 8192]
    if args.model:
        try:
            (inferred_layers, inferred_logical, inferred_hidden_sizes) = (
                infer_params_from_model(
                    model=args.model,
                )
            )
        except Exception as e:
            if rank == 0:
                print(f"Warning: failed to infer from model config: {e}")

    hidden_sizes = inferred_hidden_sizes
    num_layers = args.num_layers if args.num_layers is not None else inferred_layers
    num_logical = inferred_logical
    num_physical = saved_physical

    if saved_layers != num_layers or saved_logical != num_logical:
        raise ValueError(
            "EPLB state metadata does not match derived parameters: "
            f"saved=({saved_layers},{saved_logical}), "
            f"derived=({num_layers},{num_logical})"
        )

    num_redundant = int(saved_physical - num_logical)
    if rank == 0:
        print(
            f"Derived num_redundant_experts={num_redundant} "
            f"(from state: physical={saved_physical}, logical={num_logical})"
        )

    if num_layers <= 0:
        raise ValueError("num-layers must be provided or derivable from --model")
    if num_logical <= 0:
        raise ValueError(
            "Failed to derive number of logical experts from model or state file"
        )

    if num_physical % world_size != 0:
        raise ValueError(
            f"num_physical_experts ({num_physical})"
            f"must be divisible by world_size ({world_size})"
        )
    num_local_experts = num_physical // world_size

    # Old (initial) mapping: per-layer identical mapping
    init_row = build_initial_global_physical_to_logical_map(num_logical, num_redundant)
    old_global_map = torch.tensor(init_row, dtype=torch.int64, device=device)
    old_global_map = old_global_map.unsqueeze(0).expand(num_layers, -1).contiguous()

    # New mapping: already loaded from file once at the beginning
    new_global_map = new_map

    # Create synthetic expert weights for current placement
    expert_weights = create_expert_weights(
        num_layers=num_layers,
        num_local_experts=num_local_experts,
        hidden_sizes=hidden_sizes,
        rank=rank,
        global_mapping=old_global_map,
        dtype=dtype,
        device=device,
    )

    ep_group = dist.group.WORLD

    # Warmup GPU
    rearrange_expert_weights_inplace(
        old_global_expert_indices=old_global_map,
        new_global_expert_indices=new_global_map,
        expert_weights=expert_weights,  # type: ignore[arg-type]
        ep_group=ep_group,
        is_profile=False,
        rank_mapping=None,
    )
    torch.cuda.synchronize()
    dist.barrier()

    # Measure multiple iterations
    times: list[float] = []
    for _ in range(max(1, int(args.num_iters))):
        if rank == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        rearrange_expert_weights_inplace(
            old_global_expert_indices=old_global_map,
            new_global_expert_indices=new_global_map,
            expert_weights=expert_weights,  # type: ignore[arg-type]
            ep_group=ep_group,
            is_profile=False,
            rank_mapping=None,
        )

        dist.barrier()
        if rank == 0:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    if rank == 0 and times:
        mean_time = sum(times) / len(times)
        print(
            f"Rearrange timings (ms): mean={mean_time * 1000:.2f} | "
            f"iters={len(times)}, layers={num_layers}, logical={num_logical}, "
            f"redundant={num_redundant}, "
            f"world_size={world_size}, local_experts={num_local_experts}, "
            f"hidden_sizes={hidden_sizes}, dtype=bfloat16"
        )
    torch.distributed.destroy_process_group(ep_group)


if __name__ == "__main__":
    main()
