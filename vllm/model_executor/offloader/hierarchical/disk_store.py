# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk→RAM ExpertStore reader with O_DIRECT / thread-pool I/O."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.hierarchical.format import (
    ExpertStoreManifest,
    LayerExpertMeta,
    load_manifest,
    unpack_expert_row,
)
from vllm.v1.kv_offload.tiering.fs.io import O_DIRECT, probe_o_direct

logger = init_logger(__name__)


class ExpertStoreReader:
    """Reads expert rows from an on-disk ExpertStore into pinned uint8 buffers."""

    def __init__(
        self,
        disk_path: str,
        *,
        num_workers: int = 8,
        prefer_direct: bool = True,
    ):
        self.disk_path = Path(disk_path)
        self.prefer_direct = prefer_direct and bool(O_DIRECT)
        self._use_direct = False
        if self.prefer_direct:
            self._use_direct = probe_o_direct(str(self.disk_path))
            if not self._use_direct:
                logger.info(
                    "O_DIRECT unavailable under %s; using buffered I/O",
                    self.disk_path,
                )
        self.manifest = load_manifest(self.disk_path)
        if self.manifest is None:
            raise FileNotFoundError(
                f"ExpertStore manifest not found under {self.disk_path}"
            )
        self._layers: dict[int, LayerExpertMeta] = {
            layer.layer_id: layer for layer in self.manifest.layers
        }
        self._pool = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="expert-store"
        )
        self._lock = threading.Lock()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def has_layer(self, layer_id: int) -> bool:
        return layer_id in self._layers

    def layer_meta(self, layer_id: int) -> LayerExpertMeta:
        return self._layers[layer_id]

    def read_row_sync(self, layer_id: int, expert_id: int) -> torch.Tensor:
        """Read one expert row into a CPU uint8 tensor."""
        meta = self._layers[layer_id]
        path = self.disk_path / meta.file_name
        data = self._pread(path, expert_id * meta.row_nbytes, meta.row_nbytes)
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()

    def read_row_async(
        self, layer_id: int, expert_id: int
    ) -> Future[torch.Tensor]:
        return self._pool.submit(self.read_row_sync, layer_id, expert_id)

    def unpack_row(
        self, layer_id: int, blob: torch.Tensor
    ) -> list[torch.Tensor]:
        meta = self._layers[layer_id]
        return unpack_expert_row(blob, meta.tensor_specs)

    def _pread(self, path: Path, offset: int, nbytes: int) -> bytes:
        flags = os.O_RDONLY
        if self._use_direct:
            flags |= O_DIRECT
        # O_DIRECT requires aligned buffers; for simplicity fall back to
        # buffered reads when alignment would be painful for small rows.
        if self._use_direct and (offset % 512 != 0 or nbytes % 512 != 0):
            flags = os.O_RDONLY
        fd = os.open(str(path), flags)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            remaining = nbytes
            chunks: list[bytes] = []
            while remaining > 0:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) != nbytes:
                raise IOError(
                    f"Short read {path}@{offset}: {len(data)}/{nbytes}"
                )
            return data
        finally:
            os.close(fd)


def ensure_store_or_none(
    disk_path: str | None,
    *,
    num_workers: int,
    prefer_direct: bool,
) -> ExpertStoreReader | None:
    if not disk_path:
        return None
    path = Path(disk_path)
    if not (path / "manifest.json").exists():
        logger.warning(
            "tier_disk_path=%s has no manifest yet; disk tier inactive "
            "until ExpertStore is built",
            disk_path,
        )
        return None
    return ExpertStoreReader(
        disk_path, num_workers=num_workers, prefer_direct=prefer_direct
    )
