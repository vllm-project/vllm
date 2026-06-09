# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side of the ECCPUConnector.

Thin, stateless across steps: opens the shared mmap region and uses the
per-step connector metadata (`ECCPUConnectorMetadata`) to decide which
blocks to copy in each direction.
"""

from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer.ec_connector.cpu.metadata import (
    ECCPUConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
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
        layout = setup_ec_region(vllm_config)
        self._region = layout.region
        self._dtype = layout.dtype
        self._hidden_dim = layout.hidden_dim
        self._block_size_bytes = layout.block_size_bytes

        # Pin once from TP rank 0 — the mmap is shared across all TP
        # workers in the same process group, and cudaHostRegister must
        # not be called twice on the same address range.
        if is_pin_memory_available() and vllm_config.parallel_config.rank == 0:
            self._region.pin_memory()

        self._cpu_blocks = self._region.blocks

        # Dedicated CUDA stream keeps mmap <-> GPU copies off the model's
        # compute stream.
        self._copy_stream: torch.cuda.Stream | None = None
        self._copy_event: torch.cuda.Event | None = None

    def _ensure_copy_stream(self) -> None:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
            self._copy_event = torch.cuda.Event()

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
        src_bytes = src.reshape(-1).view(torch.uint8)
        total_bytes = src_bytes.numel()

        allocated_bytes = len(block_indices) * self._block_size_bytes
        if total_bytes > allocated_bytes:
            # EC block allocation was undersized; data will be truncated and
            # the consumer will reconstruct a wrong-shaped tensor.
            logger.error(
                "EC: encoder output truncated for mm_hash=%s: "
                "%d bytes available but only %d allocated "
                "(%d blocks × %d). shape=%s hidden_dim=%d.",
                mm_hash,
                total_bytes,
                allocated_bytes,
                len(block_indices),
                self._block_size_bytes,
                list(src.shape),
                self._hidden_dim,
            )

        self._ensure_copy_stream()
        assert self._copy_stream is not None
        with torch.cuda.stream(self._copy_stream):
            for i, block_idx in enumerate(block_indices):
                start = i * self._block_size_bytes
                end = min(start + self._block_size_bytes, total_bytes)
                if start >= end:
                    break
                self._cpu_blocks[block_idx, : end - start].copy_(
                    src_bytes[start:end], non_blocking=True
                )
        # Cross-boundary visibility: a remote consumer pulls these bytes by
        # NIXL READ, so a GPU-side event is not sufficient — the copy must be
        # CPU-complete before the blocks are served.
        self._copy_stream.synchronize()

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

        self._ensure_copy_stream()
        assert self._copy_stream is not None
        assert self._copy_event is not None

        device_type = current_platform.device_type
        with torch.cuda.stream(self._copy_stream):
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
                    gathered.view(self._dtype)
                    .reshape(len(block_indices), self._hidden_dim)
                    .to(device=device_type, non_blocking=True)
                )
            self._copy_event.record(self._copy_stream)

        torch.cuda.current_stream().wait_event(self._copy_event)

    def shutdown(self) -> None:
        # Drop the cached blocks view before region cleanup; otherwise
        # mmap_obj.close() raises BufferError on its memoryview export.
        self._cpu_blocks = None  # type: ignore[assignment]
        try:
            self._region.cleanup()
        except Exception:
            logger.debug("EC: worker region cleanup failed", exc_info=True)
