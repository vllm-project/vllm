# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager pipeline snapshots for DeepSeek V4.1 shared state."""

from dataclasses import dataclass

import torch

from .pipeline import SharingDependency


@dataclass(frozen=True, order=True)
class SharingRoute:
    kind: str
    source_layer: int
    sender: int
    receiver: int

    @property
    def key(self) -> str:
        return f"__dsv41_pp__{self.kind}.{self.source_layer}"

    @property
    def payload_keys(self) -> tuple[str, ...]:
        if self.kind in ("kv", "index_k"):
            return (f"{self.key}.ids", f"{self.key}.blocks")
        return (self.key,)


def get_sharing_routes(
    dependencies: tuple[SharingDependency, ...],
) -> tuple[SharingRoute, ...]:
    """Deduplicate consumers and relay state through every intervening stage."""
    return tuple(
        sorted(
            {
                SharingRoute(d.kind, d.source_layer, stage, stage + 1)
                for d in dependencies
                for stage in range(d.source_stage, d.consumer_stage)
            }
        )
    )


def snapshot_cache_blocks(
    cache: torch.Tensor,
    block_tables: list[torch.Tensor],
    max_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snapshot all referenced blocks, including prefix hits and quantization data.

    Scheduler-owned CPU tables keep ID selection and validation on the host.
    GPU tables retain the eager fallback, which synchronizes to select IDs.
    """
    if block_tables:
        ids = torch.cat([table.reshape(-1) for table in block_tables]).to(torch.int64)
        ids = torch.unique(ids[ids >= 0])
    else:
        ids = torch.empty(0, dtype=torch.int64, device=cache.device)
    if ids.numel() and int(ids[-1]) >= cache.shape[0]:
        raise ValueError("Pipeline snapshot refers to an unallocated cache block")
    block_bytes = cache[0].numel() * cache.element_size() if cache.shape[0] else 0
    if ids.numel() * (block_bytes + ids.element_size()) > max_bytes:
        raise ValueError("Pipeline sharing snapshot exceeds its configured byte budget")
    device_ids = ids.to(device=cache.device, non_blocking=True)
    return ids, cache.index_select(0, device_ids)


def restore_cache_blocks(
    cache: torch.Tensor, ids: torch.Tensor, blocks: torch.Tensor
) -> None:
    """Refresh a local replica using the global scheduler's physical block ids."""
    if blocks.shape != (ids.numel(), *cache.shape[1:]) or blocks.dtype != cache.dtype:
        raise ValueError("Pipeline cache replica has a different block layout")
    if ids.ndim != 1 or ids.dtype != torch.int64:
        raise ValueError("Pipeline cache block ids must be an int64 vector")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= cache.shape[0]):
        raise ValueError("Pipeline snapshot refers to an unallocated replica block")
    device_ids = ids.to(device=cache.device, non_blocking=True)
    cache.index_copy_(0, device_ids, blocks)
