# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The original PyTorch CUDA/ROCm IPC weight-sharing backend."""

from dataclasses import dataclass

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
    IPCBackend,
    TensorKind,
)


@dataclass(slots=True)
class TensorEntry:
    """A single cached tensor.

    CUDA tensors are exported as `torch.multiprocessing` reduction args
    (CUDA IPC handles); non-CUDA tensors are shipped by value.
    """

    kind: TensorKind
    ipc_args: tuple | None = None
    cpu_tensor: torch.Tensor | None = None

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, kind: TensorKind) -> "TensorEntry":
        tensor = tensor.detach()
        if tensor.is_cuda:
            _, ipc_args = reduce_tensor(tensor)
            return cls(kind=kind, ipc_args=ipc_args)
        return cls(kind=kind, cpu_tensor=tensor.cpu())

    def rebuild(
        self,
        device_index: int,
        buffers: dict[object, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.ipc_args is None:
            assert self.cpu_tensor is not None
            return self.cpu_tensor
        args = list(self.ipc_args)
        # Index 6 of the args from reduce_tensor is the device index. It must
        # be retargeted to the local index since the daemon and the engine may
        # have different CUDA_VISIBLE_DEVICES mappings.
        args[6] = device_index
        return rebuild_cuda_tensor(*args)


class CudaIpcBackend(IPCBackend):
    def export_tensor(self, tensor: torch.Tensor, kind: TensorKind) -> TensorEntry:
        return TensorEntry.from_tensor(tensor, kind)
