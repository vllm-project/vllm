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
        self.plan = plan
        descriptor = config.kv_cache_tensors[0]
        first = caches[descriptor.layers[0]]
        backing = torch.empty(0, dtype=torch.int8, device=first.device).set_(
            first.untyped_storage()
        )
        self.regions = [r for r in plan.regions if r.bundle.owner is not None]
        self.buffers = [backing.narrow(0, r.offset, r.size) for r in self.regions]
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
        self.has_history = False
        logger.info(
            "KVPP rank %d/%d (%s): %d logical blocks, %d/%d owned target bundles, "
            "%d persistent bytes, %d total allocation bytes, "
            "%d broadcast payload bytes per forward with history",
            plan.placement.rank,
            plan.placement.world_size,
            "bcast",
            config.num_blocks,
            sum(r.bundle.owner == plan.placement.rank for r in self.regions),
            len(self.regions),
            sum(r.size for r in plan.regions if r.scratch_slot is None),
            plan.backing_size,
            sum(r.size for r in self.regions),
        )

    def prepare_forward(self, has_history: bool) -> None:
        assert (
            self.next_index == len(self.regions)
            and self.active_index is None
            and self.pending is None
        ), "Previous KVPP forward is incomplete."
        self.has_history = has_history
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
            work = dist.broadcast(
                self.buffers[index],
                src=self.ranks[region.bundle.owner],
                group=self.group,
                async_op=True,
            )
            work.wait()
            done = torch.cuda.Event()
            done.record(self.transfer_stream)
        self.pending = (index, work, done)

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
        if self.active_index != index:
            raise RuntimeError(f"KVPP release without acquisition: {layer_name}.")
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


def maybe_prepare_kvpp(
    kvpp_runtime: KVPPRuntime | None, num_computed_tokens: np.ndarray
) -> None:
    if kvpp_runtime is not None:
        kvpp_runtime.prepare_forward(bool(np.any(num_computed_tokens > 0)))
