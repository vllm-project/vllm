# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize KVPP caches before their first device access."""

from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed import get_kvpp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class KVPPRuntime:
    """One ordered execution context with one-layer-ahead prefetch.

    Tensor views are fixed at initialization. Events protect the owner's
    source reads as well as the receivers' writes and scratch-buffer reuse.
    """

    def __init__(self, config: KVCacheConfig, caches: dict[str, torch.Tensor]):
        plan = config.storage_plan
        assert plan is not None
        group = get_kvpp_group()
        assert (plan.placement.rank, plan.placement.world_size) == (
            group.rank_in_group,
            group.world_size,
        ), "KVPP allocation and execution replica ranks differ."
        self.group = group.device_group
        self.ranks = group.ranks
        descriptor = config.kv_cache_tensors[0]
        first = caches[descriptor.layers[0]]
        backing = torch.empty(0, dtype=torch.int8, device=first.device).set_(
            first.untyped_storage()
        )
        self.regions = [r for r in plan.regions if r.bundle.owner is not None]
        # Per-component [num_blocks, page_bytes] views; transfers pack only the
        # blocks a batch reads through the staging buffer.
        tensors = {t.layers[0]: t for t in config.kv_cache_tensors}
        self.component_views: list[list[torch.Tensor]] = []
        for region in self.regions:
            views = []
            for name in region.bundle.layers:
                t = tensors[name]
                nbytes = t.block_stride * config.num_blocks
                assert region.offset <= t.offset <= region.offset + region.size - nbytes
                views.append(
                    backing.narrow(0, t.offset, nbytes).view(
                        config.num_blocks, t.block_stride
                    )
                )
            self.component_views.append(views)
        self.rank = plan.placement.rank
        self.staging = torch.empty(
            plan.staging_size, dtype=torch.int8, device=first.device
        )
        self.block_ids: torch.Tensor | None = None
        self.layer_indices = {
            name: index
            for index, region in enumerate(self.regions)
            for name in region.bundle.layers
        }
        self.transfer_stream = torch.cuda.Stream(device=first.device)
        self.slot_last_use: dict[int, torch.cuda.Event] = {}
        self.pending: tuple[int, Any, torch.cuda.Event] | None = None
        self.active_index: int | None = None
        self.next_index = len(self.regions)
        logger.info(
            "KVPP rank %d/%d: %d logical blocks, %d/%d owned target bundles, "
            "%d persistent bytes, %d total allocation bytes, %d staging bytes",
            plan.placement.rank,
            plan.placement.world_size,
            config.num_blocks,
            sum(r.bundle.owner == plan.placement.rank for r in self.regions),
            len(self.regions),
            sum(r.size for r in plan.regions if r.scratch_slot is None),
            plan.backing_size,
            plan.staging_size,
        )

    @property
    def has_history(self) -> bool:
        return self.block_ids is not None

    def prepare_forward(self, block_ids: np.ndarray | None) -> None:
        """Start a forward that reads ``block_ids`` of earlier KV, if any."""
        assert (
            self.next_index == len(self.regions)
            and self.active_index is None
            and self.pending is None
        ), "Previous KVPP forward is incomplete."
        self.block_ids = None
        if block_ids is not None:
            assert len(block_ids) > 0
            self.block_ids = torch.from_numpy(block_ids).to(
                self.transfer_stream.device, non_blocking=True
            )
            self.block_ids.record_stream(self.transfer_stream)
        self.next_index = 0
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream())

    def _prefetch(self, index: int) -> None:
        region = self.regions[index]
        assert region.bundle.owner is not None
        with torch.cuda.stream(self.transfer_stream):
            self.transfer_stream.wait_event(self.ready)
            if (
                region.scratch_slot is not None
                and region.scratch_slot in self.slot_last_use
            ):
                self.transfer_stream.wait_event(self.slot_last_use[region.scratch_slot])
            assert self.block_ids is not None
            work = self._broadcast_blocks(index, self.block_ids)
            done = torch.cuda.Event()
            done.record(self.transfer_stream)
        self.pending = (index, work, done)

    def _broadcast_blocks(self, index: int, block_ids: torch.Tensor) -> Any:
        """Gather, broadcast, and scatter selected blocks of one bundle.

        Runs on the transfer stream, which orders reuse of the staging buffer.
        """
        owner = self.regions[index].bundle.owner
        assert owner is not None
        views = self.component_views[index]
        widths = [view.shape[1] for view in views]
        blocks_per_chunk = self.staging.numel() // sum(widths)
        for start in range(0, block_ids.numel(), blocks_per_chunk):
            ids = block_ids[start : start + blocks_per_chunk]
            n = ids.numel()
            packed = self.staging[: n * sum(widths)]
            parts = [
                part.view(n, width)
                for part, width in zip(packed.split([n * w for w in widths]), widths)
            ]
            if owner == self.rank:
                for view, part in zip(views, parts):
                    torch.index_select(view, 0, ids, out=part)
            work = dist.broadcast(
                packed, src=self.ranks[owner], group=self.group, async_op=True
            )
            work.wait()
            if owner != self.rank:
                for view, part in zip(views, parts):
                    view.index_copy_(0, ids, part)
        return work

    def acquire(self, layer_name: str) -> None:
        index = self.layer_indices.get(layer_name)
        if index is None or index == self.active_index:
            return
        assert index == self.next_index and self.active_index is None, (
            f"Out-of-order KVPP access to {layer_name}."
        )
        if self.has_history:
            if self.pending is None:
                self._prefetch(index)
            assert self.pending is not None and self.pending[0] == index
            torch.cuda.current_stream().wait_event(self.pending[2])
            self.pending = None
        self.active_index = index
        if self.has_history and index + 1 < len(self.regions):
            self._prefetch(index + 1)

    def release(self, layer_name: str) -> None:
        index = self.layer_indices.get(layer_name)
        if index is None:
            return
        region = self.regions[index]
        # Auxiliary indexer accesses end before the bundle's main attention.
        if layer_name != region.bundle.layers[0]:
            return
        assert self.active_index == index, (
            f"KVPP release without acquisition: {layer_name}."
        )
        if region.scratch_slot is not None:
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream())
            self.slot_last_use[region.scratch_slot] = done
        self.active_index = None
        self.next_index = index + 1


def get_kvpp_runtime_cls() -> type[KVPPRuntime]:
    path = current_platform.get_kvpp_runtime_cls()
    if path is None:
        raise ValueError("The device platform has no KVPP runtime.")
    return resolve_obj_by_qualname(path)


def create_kvpp_runtime(
    config: KVCacheConfig, caches: dict[str, torch.Tensor]
) -> KVPPRuntime | None:
    if config.storage_plan is None:
        return None
    return get_kvpp_runtime_cls()(config, caches)
