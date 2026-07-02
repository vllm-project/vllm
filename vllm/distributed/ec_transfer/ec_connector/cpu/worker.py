# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side of the ECCPUConnector.

Thin, stateless across steps: opens the shared mmap region and uses the
per-step connector metadata (`ECCPUConnectorMetadata`) to decide which
blocks to copy in each direction.
"""

from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECRegionContext,
    setup_ec_region,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ECCPUWorker:
    """Worker-side delegate for the ECCPUConnector.

    - Producer role: copies `encoder_cache[mm_hash]` → `mmap[block_indices]`
      for each entry in `metadata.saves`, then synchronizes so the bytes
      are visible to the scheduler process's subsequent NIXL reads.
    - Consumer role: copies `mmap[block_indices]` → `encoder_cache[mm_hash]`
      for each entry in `metadata.loads`, ordering against the compute
      stream via a CUDA event.
    - On `ec_both` nodes both paths run back-to-back in a single step.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        # Same helper the scheduler uses; both processes converge on the
        # same mmap file via `instance_id=engine_id`.
        self._memory_context: ECRegionContext = setup_ec_region(vllm_config)

        # Each TP worker lives in its own process with its own GPU, so each
        # must register the mmap with its own GPU via cudaHostRegister.
        if is_pin_memory_available():
            self._memory_context.region.pin_memory()

        self._cpu_blocks: torch.Tensor = self._memory_context.region.blocks

        # Dedicated CUDA streams keep mmap <-> GPU copies off the compute stream.
        # Separate streams allow save and load to run concurrently on ec_both nodes
        # (they operate on disjoint mm_hashes and have no data dependency).
        # Safe to create here: set_device_index() has already run before
        # ensure_ec_transfer_initialized() is called.
        self._save_stream: torch.cuda.Stream = torch.cuda.Stream()
        self._load_stream: torch.cuda.Stream = torch.cuda.Stream()

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        """Producer path: copy `encoder_cache[mm_hash]` into the mmap at the
        block indices the scheduler pre-allocated in `meta.saves`.

        The copy runs on a dedicated stream and is followed by a stream
        sync: CUDA stream ordering does not reach a remote reader, so the
        bytes must be CPU-visible in the mmap before a consumer's NIXL READ
        can pull them.
        """
        block_indices = connector_metadata.saves.get(mm_hash)
        if block_indices is None:
            # This mm_hash is not scheduled for remote serving (either the
            # producer opted out, or we're running on an ec_consumer-only
            # node). Nothing to do.
            return

        src = encoder_cache[mm_hash]
        # View as flat bytes for indexed block-sized slicing.
        src_bytes = src.view(-1).view(torch.uint8)
        total_bytes = src_bytes.numel()

        allocated_bytes = len(block_indices) * self._memory_context.block_size_bytes
        assert total_bytes <= allocated_bytes, (
            f"EC: encoder output exceeds allocated blocks for mm_hash={mm_hash}: "
            f"{total_bytes} bytes but only {allocated_bytes} allocated "
            f"({len(block_indices)} blocks × {self._memory_context.block_size_bytes}). "
            f"shape={list(src.shape)} hidden_dim={self._memory_context.hidden_dim}"
        )

        # Wait for encodings to be generated on GPU before copying.
        self._save_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._save_stream):
            for i, block_idx in enumerate(block_indices):
                start = i * self._memory_context.block_size_bytes
                end = min(start + self._memory_context.block_size_bytes, total_bytes)
                if start >= end:
                    break
                self._cpu_blocks[block_idx, : end - start].copy_(
                    src_bytes[start:end], non_blocking=True
                )
        # The mmap→GPU reload runs on a separate stream in a later step with
        # no cross-stream event, so a GPU-side event is not sufficient — the
        # GPU→mmap copy must be CPU-complete before the block is served.
        self._save_stream.synchronize()

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        """Consumer path: copy arrived mm_hashes from mmap into
        `encoder_cache` on the GPU.

        Ordering is GPU-only: we record an event on the copy stream and
        have the current (compute) stream wait on it, so downstream
        kernels that consume `encoder_cache` see the data without a
        CPU-side synchronize.
        """
        metadata = connector_metadata
        if not metadata.loads:
            return

        device_type = current_platform.device_type
        with torch.cuda.stream(self._load_stream):
            for mm_hash, block_indices in metadata.loads.items():
                if mm_hash in encoder_cache:
                    continue
                # Advanced-index gather over int8 bytes, then re-view as
                # the model dtype so the downstream `_gather_mm_embeddings`
                # slice gets `(n_tokens, hidden_dim)` in bf16/fp16 — the
                # mmap is a flat byte buffer; the dtype lives only on the
                # producer's encoder output and must be reapplied here.
                gathered = self._cpu_blocks[block_indices]
                encoder_cache[mm_hash] = (
                    gathered.view(self._memory_context.dtype)
                    .reshape(len(block_indices), self._memory_context.hidden_dim)
                    .to(device=device_type, non_blocking=True)
                )
        torch.cuda.current_stream().wait_stream(self._load_stream)

    def shutdown(self) -> None:
        self._save_stream.synchronize()
        self._load_stream.synchronize()
        # Drop the cached blocks view before region cleanup; otherwise
        # mmap_obj.close() raises BufferError on its memoryview export.
        self._cpu_blocks = None  # type: ignore[assignment]
        try:
            self._memory_context.region.cleanup()
        except Exception:
            logger.debug("EC: worker region cleanup failed", exc_info=True)
