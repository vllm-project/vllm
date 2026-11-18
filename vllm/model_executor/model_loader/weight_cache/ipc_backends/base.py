# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The local weight-sharing backend contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

TensorKind = Literal["param", "buffer"]


@dataclass(slots=True)
class StorageRecord:
    owner: object
    storage: torch.UntypedStorage


class StorageEntry(Protocol):
    def rebuild(self, device_index: int) -> StorageRecord: ...


def storage_bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


class IPCBackendBase(ABC):
    """Materialize shared storage and export rebuildable entries."""

    def materialize(self, storages: list[torch.UntypedStorage]) -> list[StorageRecord]:
        """Return records for the canonical storages after warmup."""
        return [StorageRecord(owner=storage, storage=storage) for storage in storages]

    @abstractmethod
    def export_storage(self, record: StorageRecord) -> StorageEntry:
        """Describe a storage without copying it for each consumer."""
