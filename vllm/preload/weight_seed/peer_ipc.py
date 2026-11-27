# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `peer_ipc` weight mover: copy storages between GPUs on one host."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.preload.ipc_backends.base import rebuild_entries, record_bytes
from vllm.preload.ipc_backends.torch_ipc import TorchIPCBackend
from vllm.preload.weight_seed.core import PEER_IPC_BACKEND, WeightMover

if TYPE_CHECKING:
    from vllm.preload.ipc_backends.base import IPCBackendBase, StorageRecord


class PeerIpcMover(WeightMover):
    """Copy storages through node-local weight-sharing handles."""

    name = PEER_IPC_BACKEND

    def __init__(
        self,
        device_index: int | None = None,
        ipc_backend: "IPCBackendBase | None" = None,
    ):
        # The D2D serving thread never set a device, and an IPC handle must
        # name the daemon's device; None means the device is already current
        # (e.g. the loader's own thread).
        self._device_index = device_index
        self._ipc_backend = ipc_backend or TorchIPCBackend()

    def prepare_seed(self, storages: list["StorageRecord"]) -> dict:
        if self._device_index is not None:
            torch.accelerator.set_device_index(self._device_index)
        index = torch.accelerator.current_device_index()
        gpu_uuid = None
        if torch.accelerator.device_count() > 0 and current_platform.is_cuda_alike():
            gpu_uuid = current_platform.get_device_uuid(index)
        entries = [self._ipc_backend.export_storage(record) for record in storages]
        return {
            "backend": self.name,
            "storages": entries,
            "source_device_index": index,
            "source_gpu_uuid": gpu_uuid,
        }

    def copy_into(
        self,
        storages: list[Mapping],
        seed: Mapping,
        buffers: list[torch.Tensor],
    ) -> None:
        source_index = seed["source_device_index"]
        source_gpu_uuid = seed.get("source_gpu_uuid")
        if source_gpu_uuid is not None:
            source_index = next(
                (
                    index
                    for index in range(torch.accelerator.device_count())
                    if current_platform.get_device_uuid(index) == source_gpu_uuid
                ),
                None,
            )
            if source_index is None:
                raise RuntimeError(
                    f"Source GPU {source_gpu_uuid} is not visible to this "
                    "mirror; cannot open its local weight handle."
                )
        entries = seed["storages"]
        if len(entries) != len(storages):
            raise RuntimeError("Source daemon omitted manifest storages")
        sources = []
        try:
            sources.extend(rebuild_entries(entries, source_index))
            for source, buffer in zip(sources, buffers, strict=True):
                source_bytes = record_bytes(source, buffer.nbytes)
                buffer.copy_(source_bytes, non_blocking=True)
        finally:
            torch.accelerator.synchronize()
