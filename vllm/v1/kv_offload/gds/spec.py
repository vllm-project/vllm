# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading spec — factory for GDS manager and worker."""

from typing_extensions import override

from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.gds.manager import GDSOffloadingManager
from vllm.v1.kv_offload.gds.worker import GDSOffloadingWorker


class GDSOffloadingSpec(CPUOffloadingSpec):
    """Spec for GPU Direct Storage offloading via cuFile sync API."""

    def __init__(self, config: OffloadingConfig):
        # GDS doesn't need cpu_bytes_to_use but CPUOffloadingSpec requires it.
        extra_config = dict(config.extra_config)
        if "cpu_bytes_to_use" not in extra_config:
            extra_config["cpu_bytes_to_use"] = 1
            config = OffloadingConfig(
                groups=config.groups,
                worker_kv_bytes_per_block=config.worker_kv_bytes_per_block,
                enable_kv_cache_events=config.enable_kv_cache_events,
                extra_config=extra_config,
                engine_id=config.engine_id,
                model=config.model,
                cache=config.cache,
                parallel=config.parallel,
            )

        super().__init__(config)

        self.gds_root_dir: str = self.extra_config.get("gds_root_dir", "/tmp/vllm_gds")
        self.read_only: bool = bool(self.extra_config.get("read_only", False))
        self._gds_worker: GDSOffloadingWorker | None = None

        # Compute block_size_factor: offloaded_block_size / gpu_block_size.
        self.block_size_factor: int = 1
        offloaded_block_size = self.extra_config.get("block_size")
        if offloaded_block_size is not None:
            gpu_block_sizes = set(self.tokens_per_block)
            assert len(gpu_block_sizes) == 1, (
                "If 'block_size' is specified, all KV cache groups "
                "must have the same block size."
            )
            gpu_block_size = gpu_block_sizes.pop()
            offloaded_block_size_int = int(offloaded_block_size)
            assert offloaded_block_size_int % gpu_block_size == 0
            self.block_size_factor = offloaded_block_size_int // gpu_block_size

    def _get_file_mapper(self) -> FileMapper:
        return FileMapper.from_offloading_spec(
            root_dir=self.gds_root_dir,
            offloading_spec=self,
            blocks_per_file=self.block_size_factor,
            parallel_agnostic=True,
        )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            self._manager = GDSOffloadingManager(
                file_mapper=self._get_file_mapper(),
                enable_events=self.kv_events_config.enable_kv_cache_events,
                read_only=self.read_only,
            )
        return self._manager

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        raise NotImplementedError("GDS uses get_worker() directly")

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._gds_worker:
            if not current_platform.is_cuda_alike():
                raise RuntimeError("GDS offloading requires a CUDA-capable GPU.")
            file_mapper = self._get_file_mapper()
            self._gds_worker = GDSOffloadingWorker(
                block_size_factor=self.block_size_factor,
                file_mapper=file_mapper,
                max_io_threads=self.extra_config.get("gds_max_io_threads", 32),
            )
        return self._gds_worker
