# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize layer-sharded caches before their first device access."""

from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.kv_cache_interface import KVCacheConfig

_device_group: Any = None
logger = init_logger(__name__)


def initialize_kv_cache_transport() -> Any:
    """Initialize a separate communicator before worker memory profiling."""
    global _device_group
    if _device_group is None:
        if not current_platform.is_cuda():
            raise ValueError("KVPP broadcast currently requires NVIDIA CUDA.")
        tp = get_tp_group()
        _device_group = dist.new_group(
            ranks=tp.ranks,
            backend=dist.get_backend(tp.device_group),
            use_local_synchronization=True,
        )
        probe = torch.zeros(1, dtype=torch.uint8, device="cuda")
        dist.broadcast(probe, src=tp.ranks[0], group=_device_group)
        torch.cuda.current_stream().synchronize()
    return _device_group


def destroy_kv_cache_transport() -> None:
    global _device_group
    if _device_group is not None:
        dist.destroy_process_group(_device_group)
        _device_group = None


class KVCacheRuntime:
    """One ordered execution context with one-layer-ahead prefetch.

    Tensor views are fixed at initialization. Events protect the owner's
    source reads as well as the receivers' writes and scratch-buffer reuse.
    """

    @classmethod
    def initialize_transport(cls) -> None:
        initialize_kv_cache_transport()

    @classmethod
    def shutdown_transport(cls) -> None:
        destroy_kv_cache_transport()

    def __init__(self, config: KVCacheConfig, caches: dict[str, torch.Tensor]):
        plan = config.storage_plan
        assert plan is not None
        tp = get_tp_group()
        if (plan.placement.rank, plan.placement.world_size) != (
            tp.rank_in_group,
            tp.world_size,
        ):
            raise ValueError("KVPP allocation and execution replica ranks differ.")
        self.group = initialize_kv_cache_transport()
        self.ranks = tp.ranks
        self.plan = plan
        descriptor = config.kv_cache_tensors[0]
        first = caches[descriptor.layers[0]]
        base = first.storage_offset() * first.element_size() - descriptor.offset
        backing = torch.empty(0, dtype=torch.int8, device=first.device).set_(
            first.untyped_storage(), base, (plan.backing_size,), (1,)
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
        self.next_index = 0
        self.has_history = False
        self.running = False
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
            plan.allocation_bytes,
            sum(r.size for r in self.regions),
        )

    @contextmanager
    def forward(self, has_history: bool):
        if self.running:
            raise RuntimeError("Concurrent KVPP forwards require separate scratch.")
        context = get_forward_context()
        previous = context.kv_cache_runtime
        context.kv_cache_runtime = self
        self.running = True
        self.has_history = has_history
        self.next_index = 0
        self.active_index = None
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream())
        try:
            yield
            if self.next_index != len(self.regions) or self.active_index is not None:
                raise RuntimeError("KVPP forward did not consume every target bundle.")
        finally:
            if self.pending is not None:
                # Cover partially executed forwards before storage is released.
                self.pending[2].synchronize()
                self.pending = None
            context.kv_cache_runtime = previous
            self.running = False

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
        if index != self.next_index or self.active_index is not None:
            raise RuntimeError(f"Out-of-order KVPP access to {layer_name}.")
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


def get_kv_cache_runtime_cls() -> type[KVCacheRuntime]:
    path = current_platform.get_kv_cache_runtime_cls()
    if path is None:
        raise ValueError("The device platform has no layer-sharded KV runtime.")
    return resolve_obj_by_qualname(path)


def create_kv_cache_runtime(
    config: KVCacheConfig, caches: dict[str, torch.Tensor]
) -> KVCacheRuntime | None:
    if config.storage_plan is None:
        return None
    return get_kv_cache_runtime_cls()(config, caches)


def shutdown_kv_cache_runtime() -> None:
    if current_platform.get_kv_cache_runtime_cls() is not None:
        get_kv_cache_runtime_cls().shutdown_transport()


@contextmanager
def kv_cache_forward(runtime: KVCacheRuntime | None, has_history: bool):
    if runtime is None:
        yield
    else:
        with runtime.forward(has_history):
            yield
