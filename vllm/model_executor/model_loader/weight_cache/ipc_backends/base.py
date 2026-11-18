# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The local weight-sharing backend contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch

TensorKind = Literal["param", "buffer"]
"""Whether a cached tensor is a model parameter or a buffer."""


@dataclass
class StorageRecord:
    owner: Any
    storage: torch.UntypedStorage
    nbytes: int
    allocation_nbytes: int


class StorageEntry(Protocol):
    def rebuild(self, device: int) -> StorageRecord:
        """Import the storage on a local device."""
        ...


def storage_bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    """A flat byte tensor covering the whole storage."""
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


class IPCBackend(ABC):
    """Materialize shared storage and export rebuildable entries."""

    def materialize(self, storages: list[torch.UntypedStorage]) -> list[StorageRecord]:
        """Return records for the canonical storages after warmup."""
        return [
            StorageRecord(
                owner=storage,
                storage=storage,
                nbytes=storage.nbytes(),
                allocation_nbytes=storage.nbytes(),
            )
            for storage in storages
        ]

    @abstractmethod
    def export_storage(self, record: StorageRecord) -> StorageEntry:
        """Describe a storage without copying it for each consumer."""
