# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The original PyTorch CUDA/ROCm IPC weight-sharing backend."""

from dataclasses import dataclass

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
    IPCBackendBase,
    StorageEntry,
    StorageRecord,
    storage_bytes,
)
from vllm.model_executor.model_loader.weight_cache.ipc_backends.memfd_ipc import (
    MemfdStorageEntry,
)


@dataclass(slots=True)
class TensorEntry:
    """A single storage exported through CUDA IPC or shipped by value."""

    ipc_args: tuple | None = None
    cpu_tensor: torch.Tensor | None = None

    @classmethod
    def from_storage(cls, storage: torch.UntypedStorage) -> "TensorEntry":
        buffer = storage_bytes(storage)
        if buffer.is_cuda:
            _, ipc_args = reduce_tensor(buffer)
            return cls(ipc_args=ipc_args)
        return cls(cpu_tensor=buffer.cpu())

    def rebuild(self, device_index: int, cache: dict | None = None) -> StorageRecord:
        if self.ipc_args is None:
            assert self.cpu_tensor is not None
            buffer = self.cpu_tensor
        else:
            args = list(self.ipc_args)
            # Index 6 of the args from reduce_tensor is the device index. It must
            # be retargeted to the local index since the daemon and the engine may
            # have different CUDA_VISIBLE_DEVICES mappings.
            args[6] = device_index
            buffer = rebuild_cuda_tensor(*args)
        storage = buffer.untyped_storage()
        return StorageRecord(owner=buffer, storage=storage)


class TorchIPCBackend(IPCBackendBase):
    def materialize(self, storages: list[torch.UntypedStorage]) -> list[StorageRecord]:
        records = super().materialize(storages)
        for index in range(len(storages)):
            storage = storages[index]
            if (
                storage.device.type == "cpu"
                and storage.nbytes()
                and storage.is_pinned()
            ):
                records[index] = MemfdStorageEntry.from_storage(storage)
                # Release the old pinned allocation before copying the next table.
                storages[index] = records[index].storage
                del storage
                torch.accelerator.empty_host_cache()
        return records

    def export_storage(self, record: StorageRecord) -> StorageEntry:
        if isinstance(record.owner, MemfdStorageEntry):
            return record.owner
        return TensorEntry.from_storage(record.storage)
