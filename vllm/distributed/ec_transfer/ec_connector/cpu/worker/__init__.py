# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side of the ECCPUConnector.

Thin, stateless across steps: opens the shared mmap region and uses the
per-step connector metadata (`ECCPUConnectorMetadata`) to decide which
blocks to copy in each direction.
"""

from typing import TYPE_CHECKING

import torch

from vllm._custom_ops import swap_blocks_batch
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker.descriptor_buffers import (
    DescriptorBufferPool,
    DescriptorBuffers,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ECCPUWorker:
    """Worker-side delegate for the ECCPUConnector.

    - Producer role: copies `encoder_cache[mm_hash]` → `mmap[block_ids]`
      for each entry in `metadata.saves`. Descriptor buffers are filled
      directly in `save_caches`; the actual DMA is issued as a single
      batched call in `flush_saves`.
    - Consumer role: copies `mmap[block_ids]` → `encoder_cache[mm_hash]`
      for all entries in `metadata.loads` via a single `swap_blocks_batch`
      call on the load stream.
    - On `ec_both` nodes both paths run back-to-back in a single step.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        self._region = create_ec_shared_region(vllm_config)
        # Model dtype; used to reinterpret raw int8 blocks on load.
        self._dtype = vllm_config.model_config.dtype

        if is_pin_memory_available():
            self._region.pin_memory()

        # All TP/PCP ranks hold identical encoder output. Only one rank
        # per mmap needs to write — saves host memory bandwidth.
        # DCP is a subdivision of TP, so tp_rank==0 covers it.
        self._is_save_rank = (
            get_tensor_model_parallel_rank() == 0 and get_pcp_group().rank_in_group == 0
        )

        # Dedicated stream for async mmap→GPU loads (overlaps with compute).
        self._load_stream = current_platform.Stream()

        # Descriptor buffer pool (recycled across steps, shared by both paths).
        self._buf_pool = DescriptorBufferPool()

        # Active save buffer being filled during save_caches calls this step.
        self._save_bufs: DescriptorBuffers | None = None
        self._save_count: int = 0

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        """Fill descriptor buffers directly for batched flush."""
        if not self._is_save_rank:
            return
        block_ids = connector_metadata.saves.get(mm_hash)
        if block_ids is None:
            return

        src = encoder_cache[mm_hash]
        total_bytes = src.numel() * src.element_size()
        block_size = self._region.block_size_bytes
        allocated_bytes = len(block_ids) * block_size
        assert total_bytes <= allocated_bytes, (
            f"EC: encoder output exceeds allocated blocks for mm_hash={mm_hash}: "
            f"{total_bytes} bytes but only {allocated_bytes} allocated "
            f"({len(block_ids)} blocks × {block_size}). "
            f"shape={list(src.shape)}"
        )

        if self._save_bufs is None:
            total = sum(len(v) for v in connector_metadata.saves.values())
            self._save_bufs = self._buf_pool.acquire(total)

        assert self._save_count + len(block_ids) <= self._save_bufs.src_ptrs.numel()

        src_ptrs, dst_ptrs, sizes = self._save_bufs
        src_base = src.view(-1).view(torch.uint8).data_ptr()
        dst_base = self._region.blocks.data_ptr()
        idx = self._save_count

        for i, block_idx in enumerate(block_ids):
            start = i * block_size
            src_ptrs[idx] = src_base + start
            dst_ptrs[idx] = dst_base + block_idx * block_size
            sizes[idx] = min(block_size, total_bytes - start)
            idx += 1

        self._save_count = idx

    def flush_saves(self) -> None:
        """Flush all accumulated saves in a single swap_blocks_batch call."""
        if self._save_count == 0:
            return

        bufs = self._save_bufs
        assert bufs is not None
        src_ptrs, dst_ptrs, sizes = bufs
        n = self._save_count
        swap_blocks_batch(src_ptrs[:n], dst_ptrs[:n], sizes[:n])

        self._buf_pool.release(bufs)
        self._save_bufs = None
        self._save_count = 0

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        """Consumer path: single batched copy of all loads from mmap→GPU."""
        if not connector_metadata.loads:
            return

        block_size = self._region.block_size_bytes
        blocks = self._region.blocks
        dtype = self._dtype
        device_type = current_platform.device_type
        src_base = blocks.data_ptr()

        # Pre-filter: only hashes not already in encoder_cache.
        load_items = {
            h: idxs
            for h, idxs in connector_metadata.loads.items()
            if h not in encoder_cache
        }
        if not load_items:
            return

        total_blocks = sum(len(idxs) for idxs in load_items.values())

        with current_platform.stream(self._load_stream):
            # Single contiguous destination buffer for all loads.
            dst_buf = torch.empty(
                total_blocks, block_size, dtype=torch.int8, device=device_type
            )
            dst_buf_base = dst_buf.data_ptr()

            bufs = self._buf_pool.acquire(total_blocks)
            src_ptrs = bufs.src_ptrs[:total_blocks]
            dst_ptrs = bufs.dst_ptrs[:total_blocks]
            sizes = bufs.sizes[:total_blocks]
            sizes[:] = block_size

            op_idx = 0
            for block_ids in load_items.values():
                for block_idx in block_ids:
                    src_ptrs[op_idx] = src_base + block_idx * block_size
                    dst_ptrs[op_idx] = dst_buf_base + op_idx * block_size
                    op_idx += 1

            swap_blocks_batch(src_ptrs, dst_ptrs, sizes, is_src_access_order_any=True)

            self._buf_pool.release(bufs)

            # Slice contiguous buffer into per-hash views.
            offset = 0
            for mm_hash, block_ids in load_items.items():
                n = len(block_ids)
                encoder_cache[mm_hash] = (
                    dst_buf[offset : offset + n].view(dtype).reshape(n, -1)
                )
                offset += n

        current_platform.current_stream().wait_stream(self._load_stream)

    def shutdown(self) -> None:
        self._load_stream.synchronize()
        self._save_bufs = None
        self._save_count = 0
        try:
            self._region.cleanup()
        except Exception:
            logger.debug("EC: worker region cleanup failed", exc_info=True)
