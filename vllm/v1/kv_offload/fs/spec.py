# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec for direct device <-> filesystem offloading."""

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.factory import FSWorkerFactory
from vllm.v1.kv_offload.fs.manager import DEFAULT_INDEX_SIZE, FSOffloadingManager


class FSOffloadingSpec(OffloadingSpec):
    """Offloads KV directly between device memory and files, with the worker
    backend picked by the fs_backend key (see FSWorkerFactory)."""

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        # Fail at startup, on the scheduler too, if the backend is unknown.
        FSWorkerFactory.get_worker_cls(self.extra_config)
        root_dir = self.extra_config.get("root_dir")
        if not root_dir:
            raise ValueError("root_dir must be specified in kv_connector_extra_config")
        self.root_dir: str = root_dir
        self.index_size = int(
            self.extra_config.get("fs_index_size", DEFAULT_INDEX_SIZE)
        )
        # Same naming as the FS secondary tier, so either can read the other's
        # files.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=self,
            blocks_per_file=self.blocks_per_chunk,
            parallel_agnostic=True,
        )

        self._manager: OffloadingManager | None = None
        self._worker: OffloadingWorker | None = None

    @override
    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = FSOffloadingManager(
                file_mapper=self.file_mapper,
                index_size=self.index_size,
            )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            self._worker = FSWorkerFactory.create_worker(
                self.extra_config,
                file_mapper=self.file_mapper,
                blocks_per_file=self.blocks_per_chunk,
                device=kv_caches.tensors[0].tensor.device,
                kv_caches=kv_caches,
                root_dir=self.root_dir,
            )
        return self._worker
