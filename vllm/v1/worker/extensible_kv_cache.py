# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache backing allocation that grows after warmup."""

from bisect import bisect_right
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.extensible_tensor import (
    ExtensibleTensor,
    granule_aligned_blocks,
    granule_block_alignment,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheTensor,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.gpu.attn_utils import bind_kv_caches
from vllm.v1.worker.gpu.kv_connector import get_kv_connector
from vllm.v1.worker.utils import layer_kernel_block_size, layer_spec_for_tensor

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

# Headroom left after sizing from measured memory, beyond the measured transient
# peak, for what the allocated peak does not show: the caching allocator's
# rounding and fragmentation around it, real steps combining shapes warmup
# exercises separately (a prompt-logprobs chunk alongside a live decode, the
# rejection sampler over a full spec-decode batch), and inputs profiling only
# approximates (a multimodal encoder batch of a modality other than the one
# with the largest feature size). The share scales with the model's working
# set; the floor covers small models whose measured peak is a few hundred MiB
# while their encoder or sampler transients are not. Measured on GB200: the
# first prefill of an 8B model needed over 25% beyond its allocated peak before
# the allocator could serve it, and adversarial workloads reached up to 45%.
KV_CACHE_MARGIN_FLOOR_BYTES = 1 << 30
KV_CACHE_MARGIN_FRACTION = 0.75
# Share of the headroom warmup leaves uncommitted, on top of the profiled
# activation peak, for transients the profiling run does not exercise.
KV_CACHE_WARMUP_RESERVE_FRACTION = 0.1


class ExtensibleKVCache:
    """Growable backing allocation for the KV cache.

    Reserves address space for ``kv_cache_config.num_blocks`` and commits a
    prefix of blocks on demand. Blocks keep fixed offsets, so views over the
    reservation and captured CUDA graphs stay valid as more are committed. The
    allocation splits into segments holding ``num_blocks`` blocks each: one per
    layer for layer-outermost layouts (sized by that layer's block stride), one
    overall for block-outermost.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, device: torch.device):
        tensors = kv_cache_config.kv_cache_tensors
        assert tensors, "Extensible KV cache requires at least one KV cache tensor."
        sizes = {tensor.size for tensor in tensors}
        assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
        self.size = sizes.pop()
        self.capacity_blocks = kv_cache_config.num_blocks
        if self.capacity_blocks <= 0 or self.size % self.capacity_blocks != 0:
            raise ValueError(
                f"KV cache allocation of {self.size} bytes is not a whole number "
                f"of {self.capacity_blocks} blocks."
            )
        self.bytes_per_block = self.size // self.capacity_blocks
        self.segment_offsets, self.segment_strides = self._segments(tensors)
        self.num_committed_blocks = 0
        # Memory to keep free while committing before the final sizing, e.g.
        # the profiled activation peak the warmup steps are about to hit.
        self.reserved_headroom_bytes = 0
        self.buffer = ExtensibleTensor(
            self.size,
            device=device,
            segment_capacities=[
                self.capacity_blocks * stride for stride in self.segment_strides
            ],
        )

    def _segments(self, tensors: list[KVCacheTensor]) -> tuple[list[int], list[int]]:
        """Segment offsets and block strides tiling the allocation.

        With one block stride the segments are implied by it; with several, each
        tensor must give its layers contiguous ``num_blocks``-block regions.
        """
        block_strides = {tensor.block_stride for tensor in tensors}
        if len(block_strides) == 1:
            stride = block_strides.pop()
            segment_bytes = self.capacity_blocks * stride
            num_segments = self.size // segment_bytes
            return [i * segment_bytes for i in range(num_segments)], [
                stride
            ] * num_segments
        segments: dict[int, int] = {}
        for tensor in tensors:
            if tensor.layer_stride != self.capacity_blocks * tensor.block_stride:
                raise ValueError(
                    "KV cache tensors with different block strides need a "
                    "layer-outermost layout for the extensible KV cache."
                )
            for layer_idx in range(len(tensor.layers)):
                offset = tensor.offset + layer_idx * tensor.layer_stride
                stride = segments.setdefault(offset, tensor.block_stride)
                if stride != tensor.block_stride:
                    raise ValueError(
                        f"KV cache tensors alias offset {offset} with different "
                        "block strides."
                    )
        offsets = sorted(segments)
        strides = [segments[offset] for offset in offsets]
        end = 0
        for offset, stride in zip(offsets, strides):
            if offset != end:
                raise ValueError("KV cache tensors do not tile the allocation.")
            end += self.capacity_blocks * stride
        if end != self.size:
            raise ValueError("KV cache tensors do not tile the allocation.")
        return offsets, strides

    def allocate(self, size: int) -> torch.Tensor:
        """Allocator hook for ``allocate_kv_cache``; commits only the null block."""
        assert size == self.size, f"Expected {self.size} bytes, got {size}."
        self.commit(1)
        return self.buffer.full_view()

    def commit(
        self, num_blocks: int, defragment: bool = False, shrink: bool = False
    ) -> None:
        """Back the first ``num_blocks`` blocks; new blocks are zeroed.

        Grow-only unless ``shrink`` is set, which remaps a smaller prefix and
        discards contents. ``defragment`` remaps each segment as one driver
        allocation (discarding contents): UCX cannot RDMA a range spanning
        several allocations.
        """
        num_blocks = min(num_blocks, self.capacity_blocks)
        sizes = [num_blocks * stride for stride in self.segment_strides]
        remap = (
            defragment and not all(self.buffer.segments_backed_by_one_chunk(sizes))
        ) or (shrink and num_blocks < self.num_committed_blocks)
        if remap:
            self.buffer.release_physical()
            self.num_committed_blocks = 0
        if num_blocks <= self.num_committed_blocks:
            return
        self.buffer.resize_segments_(sizes, zero_new=True)
        self.num_committed_blocks = num_blocks
        logger.debug(
            "Committed %d of %d KV cache blocks (%.2f GiB physical).",
            num_blocks,
            self.capacity_blocks,
            self.physical_bytes / (1 << 30),
        )

    def committed_views(
        self,
        kv_cache_config: KVCacheConfig,
        layout: KVCacheLayout,
        kernel_block_sizes: list[int] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Per-layer views whose storage spans exactly the committed blocks.

        ``kv_cache_config.num_blocks`` must equal the committed count. Connectors
        derive extents from ``untyped_storage().nbytes()``, so they must not see the
        reservation. None for layouts whose layers span several segments.
        """
        if not layout.is_block_compact:
            return None
        assert kv_cache_config.num_blocks == self.num_committed_blocks
        kv_caches: dict[str, torch.Tensor] = {}
        for tensor in kv_cache_config.kv_cache_tensors:
            group_id, spec = layer_spec_for_tensor(kv_cache_config, tensor)
            if not spec.has_layer_views:
                # Laid out by the layer itself over the raw buffer; as at
                # allocation, it keeps the whole reservation.
                full_view = self.buffer.full_view()
                kv_caches.update((name, full_view) for name in tensor.layers)
                continue
            kernel_block_size = layer_kernel_block_size(
                spec, group_id, kernel_block_sizes
            )
            for layer_idx, layer_name in enumerate(tensor.layers):
                segment, offset = self._locate(
                    tensor.offset + layer_idx * tensor.layer_stride
                )
                (view,) = create_kv_cache_views(
                    self.buffer.segment_view(segment),
                    spec,
                    kv_cache_config.num_blocks,
                    layout,
                    replace(tensor, layers=[layer_name], offset=offset),
                    kernel_block_size=kernel_block_size,
                )
                kv_caches[layer_name] = view
        return kv_caches

    def committable_blocks(self) -> int:
        """Blocks that can be committed now without eating into the headroom.

        Bounds what warmup may commit: free memory plus the committed prefix,
        less the sizing margin and the larger of ``reserved_headroom_bytes`` and
        a share of the headroom, never below what is already committed nor
        above the capacity.
        """
        free_memory, _ = torch.accelerator.get_memory_info(self.buffer.device)
        headroom = free_memory + self.physical_bytes
        reserve = max(
            self.reserved_headroom_bytes,
            int(headroom * KV_CACHE_WARMUP_RESERVE_FRACTION),
        )
        margin = max(
            KV_CACHE_MARGIN_FLOOR_BYTES, int(reserve * KV_CACHE_MARGIN_FRACTION)
        )
        budget = headroom - margin - reserve
        return max(self.blocks_within(budget), self.num_committed_blocks)

    @property
    def physical_bytes(self) -> int:
        """Physically mapped bytes, including granule rounding."""
        return self.buffer.physical_bytes

    def physical_bytes_for(self, num_blocks: int) -> int:
        """Bytes a commit of ``num_blocks`` maps, each segment rounded to granules."""
        granule = self.buffer.granularity
        return sum(
            -(-num_blocks * stride // granule) * granule
            for stride in self.segment_strides
        )

    @property
    def block_alignment(self) -> int:
        """Block count multiple at which every segment ends on a granule."""
        return granule_block_alignment(self.segment_strides, self.buffer.granularity)

    def blocks_within(self, num_bytes: int, aligned: bool = False) -> int:
        """Most blocks whose physical footprint fits in ``num_bytes``.

        ``aligned`` restricts the count to multiples of ``block_alignment``, so
        that a defragmenting commit backs each segment with one chunk exactly.
        """
        num_blocks = min(num_bytes // self.bytes_per_block, self.capacity_blocks)
        if aligned:
            return granule_aligned_blocks(
                num_blocks, self.segment_strides, self.buffer.granularity
            )
        while num_blocks > 0 and self.physical_bytes_for(num_blocks) > num_bytes:
            num_blocks -= 1
        return num_blocks

    def _locate(self, start: int) -> tuple[int, int]:
        """Segment index and offset within it of allocation byte ``start``."""
        segment = bisect_right(self.segment_offsets, start) - 1
        return segment, start - self.segment_offsets[segment]

    def committed_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, layout: KVCacheLayout, num_blocks: int
    ) -> list[KVCacheTensor]:
        """Tensor placements matching ``committed_views`` for ``num_blocks``.

        ``size`` is the committed total, as every tensor describes the one
        allocation. Under block-outermost layouts the committed blocks are one
        prefix of it, so the capacity placements still hold. Under layer-outermost
        layouts each layer's view has its own storage over its committed prefix:
        describe it as a one-layer tensor placed at the start of that storage.
        """
        size = num_blocks * self.bytes_per_block
        tensors = [
            replace(tensor, size=size) for tensor in kv_cache_config.kv_cache_tensors
        ]
        if layout.is_block_outermost:
            return tensors
        committed: list[KVCacheTensor] = []
        for tensor in tensors:
            _, spec = layer_spec_for_tensor(kv_cache_config, tensor)
            for layer_idx, layer_name in enumerate(tensor.layers):
                start = tensor.offset + layer_idx * tensor.layer_stride
                segment, offset = self._locate(start)
                if not spec.has_layer_views:
                    # The layer keeps the whole reservation as its storage.
                    offset = start
                committed.append(
                    KVCacheTensor(
                        size=size,
                        layers=[layer_name],
                        layer_stride=num_blocks * tensor.block_stride,
                        block_stride=tensor.block_stride,
                        offset=offset,
                    )
                )
        return committed

    def release_physical(self) -> None:
        """Drop physical pages for sleep; addresses and views stay valid."""
        self.buffer.release_physical()

    def recommit(self) -> None:
        """Map fresh zeroed pages for the blocks committed before release."""
        num_blocks = self.num_committed_blocks
        self.num_committed_blocks = 0
        self.commit(num_blocks)

    def free(self) -> None:
        self.buffer.free()
        self.num_committed_blocks = 0


def num_committable_kv_blocks(runner: "GPUModelRunner") -> int:
    """Blocks warmup may address: all of them, or, for an extensible cache,
    those whose commit still leaves the reserved headroom free.

    The count is agreed across ranks: it sets the shapes warmup runs and
    whether a step runs at all, and a rank skipping a step others take part
    in would leave them waiting in a collective.
    """
    kv_cache = getattr(runner, "extensible_kv_cache", None)
    if kv_cache is not None:
        return _min_across_ranks(kv_cache.committable_blocks())
    return runner.kv_cache_config.num_blocks


def _min_across_ranks(value: int) -> int:
    from vllm.distributed.parallel_state import get_world_group

    if not torch.distributed.is_initialized():
        return value
    world = get_world_group()
    if world.world_size == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor, group=world.cpu_group, op=torch.distributed.ReduceOp.MIN
    )
    return int(tensor.item())


def ensure_kv_cache_blocks(runner: "GPUModelRunner", num_blocks: int) -> None:
    """Commit enough of an extensible KV cache to address ``num_blocks``."""
    kv_cache = getattr(runner, "extensible_kv_cache", None)
    if kv_cache is not None:
        kv_cache.commit(num_blocks)


def extend_kv_cache(runner: "GPUModelRunner", num_blocks: int) -> None:
    """Commit the final size once post-warmup memory is known.

    Afterwards the runner matches one that allocated ``num_blocks`` up front:
    views are rebuilt over the committed bytes (same addresses) and the KV
    connector is created against final memory.
    """
    kv_cache = runner.extensible_kv_cache
    assert kv_cache is not None
    if num_blocks < kv_cache.num_committed_blocks:
        # Warmup committed more than the measured budget fits (its transient
        # peak exceeded the reserve); the contents are warmup garbage.
        logger.info(
            "Remapping the KV cache from %d to %d blocks to fit the measured budget.",
            kv_cache.num_committed_blocks,
            num_blocks,
        )
    kv_cache.commit(
        num_blocks,
        defragment=runner.vllm_config.kv_transfer_config is not None,
        shrink=True,
    )
    runner.kv_cache_config.num_blocks = kv_cache.num_committed_blocks
    logger.info(
        "Extended KV cache to %d blocks (%.2f GiB).",
        kv_cache.num_committed_blocks,
        kv_cache.physical_bytes / (1 << 30),
    )
    forward_context = runner.compilation_config.static_forward_context
    kv_caches = kv_cache.committed_views(
        runner.kv_cache_config,
        runner.cache_config.get_resolved_kv_cache_layout(),
        runner.kernel_block_sizes,
    )
    if kv_caches is not None:
        bind_kv_caches(
            kv_caches, forward_context, runner.kv_cache_config, runner.vllm_config
        )
        runner.kv_caches = [
            cache for cache in kv_caches.values() if cache.device == runner.device
        ]
    else:
        kv_caches = {
            layer_name: layer.kv_cache
            for layer_name, layer in forward_context.items()
            if isinstance(getattr(layer, "kv_cache", None), torch.Tensor)
        }
    runner.kv_connector = get_kv_connector(runner.vllm_config, kv_caches)


def measure_kv_cache_bytes(
    *,
    init_free_memory: int,
    free_memory: int,
    committed_bytes: int,
    requested_memory: int,
    transient_peak_bytes: int = 0,
    margin_floor_bytes: int = KV_CACHE_MARGIN_FLOOR_BYTES,
    margin_fraction: float = KV_CACHE_MARGIN_FRACTION,
) -> int:
    """Bytes the KV cache may occupy given the memory measured after warmup.

    Everything resident except the committed KV prefix is needed by the engine;
    the cache gets the rest of the budget, capped by the device headroom (free
    memory plus the committed prefix) less a margin of
    ``max(margin_floor_bytes, margin_fraction * transient_peak_bytes)``. The
    measured ``transient_peak_bytes`` are then kept free in full. The margin
    guards physical memory, so it never reduces an explicit budget that already
    leaves that much free.
    """
    non_kv_used = init_free_memory - free_memory - committed_bytes
    headroom = free_memory + committed_bytes
    margin = max(margin_floor_bytes, int(transient_peak_bytes * margin_fraction))
    available = min(requested_memory - non_kv_used, headroom - margin)
    available -= transient_peak_bytes
    return max(available, 0)
