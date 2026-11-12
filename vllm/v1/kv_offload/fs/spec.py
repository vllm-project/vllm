# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec for direct device <-> filesystem offloading."""

from typing_extensions import override

from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.manager import DEFAULT_INDEX_SIZE, FSOffloadingManager

# Backend name -> FSOffloadingWorker qualname, resolved only when selected.
# Each backend is constructed as
# cls(file_mapper=, block_size_factor=, device=, extra_config=).
_FS_BACKENDS: dict[str, str] = {
    "gds": "vllm.v1.kv_offload.fs.gds.worker.GDSOffloadingWorker",
}


class FSOffloadingSpec(OffloadingSpec):
    """Offloads KV directly between device memory and files, with the worker
    backend picked by the fs_backend key."""

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        self.backend: str = self.extra_config.get("fs_backend", "gds")
        if self.backend not in _FS_BACKENDS:
            raise ValueError(
                f"Unknown fs_backend {self.backend!r}; "
                f"expected one of {sorted(_FS_BACKENDS)}"
            )
        root_dir = self.extra_config.get("root_dir")
        if not root_dir:
            raise ValueError("root_dir must be specified in kv_connector_extra_config")
        self.read_only: bool = bool(self.extra_config.get("read_only", True))
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
                read_only=self.read_only,
            )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            worker_cls = resolve_obj_by_qualname(_FS_BACKENDS[self.backend])
            self._worker = worker_cls(
                file_mapper=self.file_mapper,
                block_size_factor=self.blocks_per_chunk,
                device=kv_caches.tensors[0].tensor.device,
                extra_config=self.extra_config,
            )
        return self._worker
