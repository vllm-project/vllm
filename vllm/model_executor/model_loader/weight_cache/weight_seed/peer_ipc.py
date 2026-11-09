# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `peer_ipc` weight mover: copy storages through CUDA IPC on one host."""

from collections.abc import Mapping

import torch

from vllm.model_executor.model_loader.weight_cache.protocol import TensorEntry
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    PEER_IPC_BACKEND,
    WeightMover,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    storage_bytes,
)
from vllm.platforms import current_platform


class PeerIpcMover(WeightMover):
    """Copy storages through CUDA IPC handles on the same host."""

    name = PEER_IPC_BACKEND

    def __init__(self, device_index: int | None = None):
        # The D2D serving thread never set a device, and an IPC handle must
        # name the daemon's device; None means the device is already current
        # (e.g. the loader's own thread).
        self._device_index = device_index

    def prepare_seed(self, storages: list[torch.UntypedStorage]) -> dict:
        if self._device_index is not None:
            torch.accelerator.set_device_index(self._device_index)
        index = torch.accelerator.current_device_index()
        gpu_uuid = None
        if torch.accelerator.device_count() > 0 and current_platform.is_cuda_alike():
            gpu_uuid = current_platform.get_device_uuid(index)
        return {
            "backend": self.name,
            "storages": [
                TensorEntry.from_tensor(storage_bytes(storage), "buffer")
                for storage in storages
            ],
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
                    "mirror; cannot open its CUDA IPC handle."
                )
        entries = seed["storages"]
        if len(entries) != len(storages):
            raise RuntimeError("Source daemon omitted manifest storages")
        sources = []
        try:
            for entry, buffer in zip(entries, buffers):
                source = entry.rebuild(source_index)
                sources.append(source)
                buffer.copy_(source, non_blocking=True)
        finally:
            torch.accelerator.synchronize()
