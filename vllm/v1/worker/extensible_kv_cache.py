# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache backing allocation that grows after warmup."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.extensible_tensor import ExtensibleTensor
from vllm.v1.kv_cache_interface import KVCacheConfig

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

    def commit(self, num_blocks: int) -> None:
        """Back the first ``num_blocks`` blocks; new blocks are zeroed, never shrinks."""
        num_blocks = min(num_blocks, self.capacity_blocks)
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
    """Commit the final KV cache size once post-warmup memory is known."""
    kv_cache = runner.extensible_kv_cache
    assert kv_cache is not None
    kv_cache.commit(num_blocks)
    logger.info(
        "Extended KV cache to %d blocks (%.2f GiB).",
        kv_cache.num_committed_blocks,
        kv_cache.physical_bytes / (1 << 30),
    )


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
