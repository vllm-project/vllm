#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a static EPLB mapping from expert-load statistics."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from vllm.distributed.eplb.policy.default import DefaultEplbPolicy


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if record.get("record_type") == "eplb_load_stats":
                records.append(record)
    if not records:
        raise ValueError(f"{path} does not contain eplb_load_stats records.")
    return records


def _filter_records(
    records: list[dict[str, Any]],
    start_step: int | None,
    end_step: int | None,
) -> list[dict[str, Any]]:
    filtered = []
    for record in records:
        step = int(record["step"])
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        filtered.append(record)
    if not filtered:
        raise ValueError("Step filters removed all eplb_load_stats records.")
    return filtered


def _record_logical_load(record: dict[str, Any]) -> torch.Tensor:
    expert_load = torch.tensor(record["expert_load"], dtype=torch.float32)
    p2l_map = torch.tensor(record["p2l_map"], dtype=torch.long)
    if expert_load.shape != p2l_map.shape:
        raise ValueError(
            f"expert_load shape {tuple(expert_load.shape)} does not match "
            f"p2l_map shape {tuple(p2l_map.shape)} for step {record.get('step')}."
        )
    num_layers = expert_load.shape[0]
    num_logical = int(p2l_map.max().item()) + 1
    logical_load = torch.zeros(
        (num_layers, num_logical), dtype=torch.float32, device=expert_load.device
    )
    logical_load.scatter_add_(dim=1, index=p2l_map, src=expert_load)
    return logical_load


def aggregate_logical_load(records: list[dict[str, Any]]) -> torch.Tensor:
    logical_load = _record_logical_load(records[0])
    for record in records[1:]:
        current = _record_logical_load(record)
        if current.shape != logical_load.shape:
            raise ValueError(
                f"Inconsistent logical load shape {tuple(current.shape)} "
                f"for step {record.get('step')}; expected {tuple(logical_load.shape)}."
            )
        logical_load += current
    return logical_load


def _rank_load_for_mapping(
    logical_load: torch.Tensor,
    physical_to_logical: torch.Tensor,
    num_ranks: int,
) -> torch.Tensor:
    num_layers, num_slots = physical_to_logical.shape
    if num_slots % num_ranks != 0:
        raise ValueError(
            f"num_slots={num_slots} must be divisible by num_ranks={num_ranks}."
        )
    slots_per_rank = num_slots // num_ranks
    replica_count = torch.zeros_like(logical_load)
    replica_count.scatter_add_(
        dim=1,
        index=physical_to_logical,
        src=torch.ones_like(physical_to_logical, dtype=logical_load.dtype),
    )
    per_replica_load = logical_load / replica_count.clamp_min(1)
    rank_load = torch.zeros((num_layers, num_ranks), dtype=torch.float32)
    for rank in range(num_ranks):
        start = rank * slots_per_rank
        end = start + slots_per_rank
        rank_load[:, rank] = torch.gather(
            per_replica_load, dim=1, index=physical_to_logical[:, start:end]
        ).sum(dim=1)
    return rank_load


def _balancedness(rank_load: torch.Tensor) -> float:
    avg_tokens = rank_load.mean(dim=1).sum()
    max_tokens = rank_load.max(dim=1).values.sum()
    if max_tokens.item() == 0:
        return 0.0
    return float((avg_tokens / max_tokens).item())


def _imbalance(rank_load: torch.Tensor) -> float:
    balancedness = _balancedness(rank_load)
    if balancedness == 0:
        return 0.0
    return 1.0 / balancedness


def _initial_mapping(
    num_layers: int,
    num_logical: int,
    num_redundant_experts: int,
) -> torch.Tensor:
    base = torch.arange(num_logical + num_redundant_experts, dtype=torch.long)
    base = base.remainder(num_logical)
    return base.unsqueeze(0).expand(num_layers, -1).contiguous()


def build_static_mapping(
    logical_load: torch.Tensor,
    num_redundant_experts: int,
    num_ranks: int,
    num_groups: int,
    num_nodes: int,
) -> torch.Tensor:
    num_slots = logical_load.shape[1] + num_redundant_experts
    return DefaultEplbPolicy.rebalance_experts(
        weight=logical_load,
        num_replicas=num_slots,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_ranks=num_ranks,
    ).long()


def write_mapping_record(
    output_path: Path,
    physical_to_logical: torch.Tensor,
    num_redundant_experts: int,
) -> None:
    record: dict[str, Any] = {
        "record_type": "eplb_initial_mapping",
        "version": 1,
        "num_redundant_experts": int(num_redundant_experts),
        "num_slots": int(physical_to_logical.shape[1]),
        "initial_global_assignments": {
            str(layer): [int(x) for x in physical_to_logical[layer].tolist()]
            for layer in range(physical_to_logical.shape[0])
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = _Parser(
        description="Generate a static EPLB mapping JSONL from expert-load stats.",
    )
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-redundant-experts", type=int, required=True)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=None,
        help="Number of EP ranks. Defaults to the num_ranks field in the stats.",
    )
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    args = parser.parse_args()

    if args.num_redundant_experts < 0:
        parser.error("--num-redundant-experts must be non-negative.")
    if args.num_nodes <= 0 or args.num_groups <= 0:
        parser.error("--num-nodes and --num-groups must be positive.")

    records = _filter_records(
        _load_jsonl(args.stats_path), args.start_step, args.end_step
    )
    logical_load = aggregate_logical_load(records)
    num_ranks = args.num_ranks or int(records[0]["num_ranks"])
    if num_ranks <= 0:
        parser.error("--num-ranks must be positive.")

    before_mapping = _initial_mapping(
        logical_load.shape[0], logical_load.shape[1], args.num_redundant_experts
    )
    after_mapping = build_static_mapping(
        logical_load,
        args.num_redundant_experts,
        num_ranks,
        args.num_groups,
        args.num_nodes,
    )
    before = _imbalance(_rank_load_for_mapping(logical_load, before_mapping, num_ranks))
    after = _imbalance(_rank_load_for_mapping(logical_load, after_mapping, num_ranks))
    write_mapping_record(
        args.output,
        after_mapping,
        args.num_redundant_experts,
    )

    print(f"Loaded {len(records)} stats records from {args.stats_path}")
    print(
        f"Generated mapping for {logical_load.shape[0]} layers, "
        f"{logical_load.shape[1]} logical experts, "
        f"{after_mapping.shape[1]} physical slots, {num_ranks} ranks"
    )
    print(f"Imbalance before: {before:.4f}x")
    print(f"Imbalance after:  {after:.4f}x")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
