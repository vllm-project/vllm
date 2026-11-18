# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The original PyTorch CUDA/ROCm IPC weight-sharing backend."""

from dataclasses import dataclass

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
    IPCBackend,
    StorageEntry,
    StorageRecord,
    TensorKind,
    storage_bytes,
)


@dataclass(slots=True)
class TensorEntry:
    """A single storage exported through CUDA IPC or shipped by value."""

    ipc_args: tuple | None = None
    cpu_tensor: torch.Tensor | None = None

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, kind: TensorKind) -> "TensorEntry":
        return cls.from_storage(tensor.untyped_storage())

    @classmethod
    def from_storage(cls, storage: torch.UntypedStorage) -> "TensorEntry":
        buffer = storage_bytes(storage)
        if buffer.is_cuda:
            _, ipc_args = reduce_tensor(buffer)
            return cls(ipc_args=ipc_args)
        return cls(cpu_tensor=buffer.cpu())

    def rebuild(self, device_index: int) -> StorageRecord:
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
        return StorageRecord(
            owner=buffer,
            storage=storage,
            nbytes=storage.nbytes(),
            allocation_nbytes=storage.nbytes(),
        )


class CudaIpcBackend(IPCBackend):
    def export_storage(self, record: StorageRecord) -> StorageEntry:
        return TensorEntry.from_storage(record.storage)
