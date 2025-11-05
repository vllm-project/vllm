
# ABOUTME: Runs the EPS union masking pre-pass over block tables.
# ABOUTME: Produces counters and mutates block tables prior to attention.

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from vllm.v1.eps.config import EpsRuntimeConfig
from vllm.v1.eps.context import EpsForwardContext
from vllm.v1.eps.block_filter import apply_union_mask
from vllm.v1.eps.selector import select_union_groups
from vllm.v1.eps.telemetry import (
    EpsStepCounters,
    blocks_to_groups,
    collect_unique_blocks,
)
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.worker.block_table import BlockTable


def _collect_group_energies(
    *,
    row: np.ndarray,
    cfg: EpsRuntimeConfig,
    req_runtime,
    layer_index: int,
) -> dict[int, float]:
    energy: dict[int, float] = {}
    state = req_runtime.state
    if state is None:
        return energy

    for block_offset, block_id in enumerate(row):
        group_id = block_offset // cfg.group_blocks
        logical_idx = req_runtime.block_mapping.get(int(block_id))
        if logical_idx is None:
            continue
        summary_group = logical_idx // cfg.group_blocks
        if summary_group >= state.G.shape[2]:
            energy[group_id] = float("inf")
            continue
        if torch is None:
            energy[group_id] = float("inf")
            continue
        if torch is None:
            energy[group_id] = float("inf")
            continue
        frob = state.frob2[layer_index, :, summary_group]
        if frob.numel() == 0:
            magnitude = 0.0
        else:
            magnitude = float(torch.sqrt(torch.max(frob)).item())
        energy[group_id] = magnitude

    return energy


def run_union_prepass(
    block_tables: Sequence[BlockTable],
    kv_specs: Sequence[object],
    kv_group_specs: Sequence[object],
    cfg: EpsRuntimeConfig,
    num_reqs: int,
    eps_ctx: EpsForwardContext | None,
) -> EpsStepCounters:
    counters = EpsStepCounters()
    if num_reqs <= 0 or not cfg.enabled or cfg.method == "off":
        return counters

    for group_idx, (block_table, kv_spec) in enumerate(zip(block_tables, kv_specs)):
        if isinstance(kv_spec, EncoderOnlyAttentionSpec):
            continue
        counters.layers += 1
        before_counts = block_table.num_blocks_per_row[:num_reqs].copy()
        if before_counts.sum() == 0:
            continue

        before_rows = np.array(block_table.block_table.np[:num_reqs], copy=True)
        group_ctx = None if eps_ctx is None else eps_ctx.group_runtimes[group_idx]
        layer_names = getattr(kv_group_specs[group_idx], "layer_names", [])
        layer_name = layer_names[0] if layer_names else None
        layer_index = 0
        if eps_ctx is not None and layer_name in eps_ctx.layer_map:
            layer_index = eps_ctx.layer_map[layer_name].layer_index

        for row_idx, before_blocks in enumerate(before_counts):
            before_blocks = int(before_blocks)
            if before_blocks == 0:
                continue

            total_groups = blocks_to_groups(before_blocks, cfg.group_blocks)
            groups_by_recency = list(range(total_groups - 1, -1, -1))

            if group_ctx is None:
                visit_groups = set(groups_by_recency[: cfg.last_n])
            else:
                req_runtime = group_ctx.request_runtimes[row_idx]
                energy_by_group = _collect_group_energies(
                    row=before_rows[row_idx],
                    cfg=cfg,
                    req_runtime=req_runtime,
                    layer_index=layer_index,
                )
                visit_groups = select_union_groups(
                    cfg=cfg,
                    groups_by_recency=groups_by_recency,
                    energy_by_group=energy_by_group,
                )

            apply_union_mask(
                block_table=block_table,
                row_idx=row_idx,
                visit_groups=visit_groups,
                group_blocks=cfg.group_blocks,
                sentinel=cfg.sentinel,
            )

        after_counts = block_table.num_blocks_per_row[:num_reqs].copy()
        after_rows = block_table.block_table.np[:num_reqs]

        counters.blocks_total += int(before_counts.sum())
        counters.blocks_kept += int(after_counts.sum())
        counters.groups_total += sum(
            blocks_to_groups(int(b), cfg.group_blocks) for b in before_counts
        )
        counters.groups_kept += sum(
            blocks_to_groups(int(b), cfg.group_blocks) for b in after_counts
        )

        before_unique = collect_unique_blocks(before_rows, before_counts)
        after_unique = collect_unique_blocks(after_rows, after_counts)

        counters.unique_blocks_total += len(before_unique)
        counters.unique_blocks_kept += len(after_unique)

        bytes_per_block = getattr(kv_spec, "page_size_bytes", 0)
        counters.kv_bytes_total += len(before_unique) * bytes_per_block
        counters.kv_bytes_kept += len(after_unique) * bytes_per_block

    counters.pages_total = counters.groups_total
    counters.pages_visited = counters.groups_kept
    counters.pages_skipped = counters.groups_dropped
    counters.pages_unique_total = counters.groups_total
    counters.pages_unique_kept = counters.groups_kept

    return counters

