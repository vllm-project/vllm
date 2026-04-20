# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileOffloadingSpec: File-based KV cache offloading implementation.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.file.handler import FileOffloadingHandler
from vllm.v1.kv_offload.file.load_store_spec import FileLoadStoreSpec
from vllm.v1.kv_offload.file.manager import FileOffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class FileOffloadingSpec(OffloadingSpec):
    """
    File-based offloading spec.

    Configuration options (via kv_connector_extra_config):
        - storage_dir: Directory to store KV cache files
          (default: /tmp/vllm_offload)
        - num_blocks: Maximum number of blocks to store
        - block_size_bytes: Size of each block in bytes
          (computed from KV cache config if not specified)
        - enable_events: Enable offloading events for debugging
        - num_threads: Number of threads for file I/O
          (default: 4)
    """

    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config, kv_cache_config)

        # Get storage directory
        self.storage_dir = self.extra_config.get("storage_dir", "/tmp/vllm_offload")

        # Get number of blocks
        self.num_blocks = self.extra_config.get("num_blocks")
        if self.num_blocks is None:
            raise ValueError(
                "num_blocks must be specified in kv_connector_extra_config "
                "for FileOffloadingSpec"
            )

        # Compute block size if not specified
        self.block_size_bytes = self.extra_config.get("block_size_bytes")
        if self.block_size_bytes is None:
            # Compute from KV cache config
            page_sizes = {
                kv_cache_group.kv_cache_spec.page_size_bytes
                for kv_cache_group in kv_cache_config.kv_cache_groups
            }
            if len(page_sizes) != 1:
                raise ValueError(
                    "Cannot compute block_size_bytes: multiple page sizes found"
                )
            page_size = page_sizes.pop()
            kv_bytes_per_block = (
                page_size
                * len(kv_cache_config.kv_cache_tensors)
                * vllm_config.parallel_config.world_size
            )
            self.block_size_bytes = kv_bytes_per_block * self.block_size_factor

        # Number of I/O threads for file operations
        self.num_threads = self.extra_config.get("num_threads", 4)

        # Enable offloading events
        kv_events_config = vllm_config.kv_events_config
        self.enable_events = (
            kv_events_config is not None and kv_events_config.enable_kv_cache_events
        )

        # Manager instance (scheduler-side)
        self._manager: OffloadingManager | None = None

        # Handler instance (worker-side)
        self._handler: FileOffloadingHandler | None = None

        logger.info(
            "FileOffloadingSpec: storage_dir=%s, num_blocks=%d, block_size_bytes=%d",
            self.storage_dir,
            self.num_blocks,
            self.block_size_bytes,
        )

    def get_manager(self) -> OffloadingManager:
        """Get the offloading manager (scheduler-side)."""
        if self._manager is None:
            self._manager = FileOffloadingManager(
                storage_dir=self.storage_dir,
                num_blocks=self.num_blocks,
                block_size_bytes=self.block_size_bytes,
                enable_events=self.enable_events,
            )
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """
        Get offloading handlers for GPU ↔ FILE transfers.

        Yields:
            Tuples of (src_type, dst_type, handler).
        """
        if not current_platform.is_cuda_alike():
            raise Exception(
                "File-based offloading is currently only supported on CUDA-alike GPUs"
            )

        if self._handler is None:
            self._handler = FileOffloadingHandler(
                gpu_tensors=[t.tensor for t in kv_caches.tensors],
                block_size_bytes=self.block_size_bytes,
                num_threads=self.num_threads,
            )

        assert self._handler is not None

        # GPU → FILE (offload)
        yield GPULoadStoreSpec, FileLoadStoreSpec, self._handler

        # FILE → GPU (restore)
        yield FileLoadStoreSpec, GPULoadStoreSpec, self._handler
