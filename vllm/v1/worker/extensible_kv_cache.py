# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache backing allocation that grows after warmup."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.extensible_tensor import ExtensibleTensor
from vllm.v1.kv_cache_interface import KVCacheConfig, create_kv_cache_views
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.gpu.attn_utils import bind_kv_caches
from vllm.v1.worker.gpu.kv_connector import get_kv_connector
from vllm.v1.worker.utils import layer_kernel_block_size, layer_spec_for_tensor

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

# Headroom kept free after sizing the cache from measured memory, covering
# allocations that only happen after warmup (allocator fragmentation, shapes
# warmup did not exercise, workspaces that grow at runtime).
EXTENSIBLE_KV_CACHE_MARGIN_BYTES = 150 * (1 << 20)


class ExtensibleKVCache:
    """Growable backing allocation for the KV cache.

    Reserves address space for ``kv_cache_config.num_blocks`` and commits a
    prefix of blocks on demand. Blocks keep fixed offsets, so views over the
    reservation and captured CUDA graphs stay valid as more are committed. The
    allocation splits into equal segments of ``num_blocks`` blocks each: one per
    layer for layer-outermost layouts, one overall for block-outermost.
    """

    def __init__(self, kv_cache_config: KVCacheConfig, device: torch.device):
        tensors = kv_cache_config.kv_cache_tensors
        assert tensors, "Extensible KV cache requires at least one KV cache tensor."
        sizes = {tensor.size for tensor in tensors}
        block_strides = {tensor.block_stride for tensor in tensors}
        assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
        if len(block_strides) != 1:
            raise ValueError(
                "The extensible KV cache requires every KV cache tensor to place "
                f"blocks at the same stride, got {sorted(block_strides)}."
            )
        self.size = sizes.pop()
        self.block_stride = block_strides.pop()
        self.capacity_blocks = kv_cache_config.num_blocks
        segment_bytes = self.capacity_blocks * self.block_stride
        if segment_bytes <= 0 or self.size % segment_bytes != 0:
            raise ValueError(
                f"KV cache allocation of {self.size} bytes is not a whole number "
                f"of {self.capacity_blocks}-block segments of {segment_bytes} bytes."
            )
        self.bytes_per_block = self.size // self.capacity_blocks
        self.num_committed_blocks = 0
        self.buffer = ExtensibleTensor(
            self.size, device=device, num_segments=self.size // segment_bytes
        )

    def allocate(self, size: int) -> torch.Tensor:
        """Allocator hook for ``allocate_kv_cache``; commits only the null block."""
        assert size == self.size, f"Expected {self.size} bytes, got {size}."
        self.commit(1)
        return self.buffer.full_view()

    def commit(self, num_blocks: int, defragment: bool = False) -> None:
        """Back the first ``num_blocks`` blocks; new blocks are zeroed, never shrinks.

        ``defragment`` remaps each segment as one driver allocation (discarding
        contents): UCX cannot RDMA a range spanning several allocations.
        """
        num_blocks = min(num_blocks, self.capacity_blocks)
        if defragment and self.buffer.num_physical_chunks > self.buffer.num_segments:
            self.buffer.release_physical()
            self.num_committed_blocks = 0
        if num_blocks <= self.num_committed_blocks:
            return
        self.buffer.resize_per_segment_(num_blocks * self.block_stride, zero_new=True)
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
        segment_capacity = self.buffer.segment_capacity_bytes
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
                start = tensor.offset + layer_idx * tensor.layer_stride
                segment, offset = divmod(start, segment_capacity)
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

    @property
    def physical_bytes(self) -> int:
        """Physically mapped bytes, including granule rounding."""
        return self.buffer.physical_bytes

    @property
    def commit_rounding_overhead(self) -> int:
        """Upper bound on granule rounding for any commit."""
        return self.buffer.num_segments * self.buffer.granularity

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


def ensure_kv_cache_blocks(runner: "GPUModelRunner", num_blocks: int) -> None:
    """Commit enough of an extensible KV cache to address ``num_blocks``."""
    if runner.extensible_kv_cache is not None:
        runner.extensible_kv_cache.commit(num_blocks)


def extend_kv_cache(runner: "GPUModelRunner", num_blocks: int) -> None:
    """Commit the final size once post-warmup memory is known.

    Afterwards the runner matches one that allocated ``num_blocks`` up front:
    views are rebuilt over the committed bytes (same addresses) and the KV
    connector is created against final memory.
    """
    kv_cache = runner.extensible_kv_cache
    assert kv_cache is not None
    kv_cache.commit(
        num_blocks, defragment=runner.vllm_config.kv_transfer_config is not None
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


def measure_kv_cache_blocks(
    *,
    init_free_memory: int,
    free_memory: int,
    committed_bytes: int,
    requested_memory: int,
    bytes_per_block: int,
    margin_bytes: int,
) -> int:
    """Blocks that fit in the memory measured after warmup.

    Everything resident except the committed KV prefix is needed by the engine;
    the cache gets the rest of the budget, capped by free memory, less
    ``margin_bytes``.
    """
    non_kv_used = init_free_memory - free_memory - committed_bytes
    available = min(requested_memory - non_kv_used, free_memory + committed_bytes)
    return max((available - margin_bytes) // bytes_per_block, 0)
