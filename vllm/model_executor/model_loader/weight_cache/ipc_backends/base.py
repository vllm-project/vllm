# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The local weight-sharing backend contract."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

TensorKind = Literal["param", "buffer"]


@dataclass(slots=True)
class StorageRecord:
    owner: object
    storage: torch.UntypedStorage
    byte_offset: int = 0


class StorageEntry(Protocol):
    def rebuild(
        self, device_index: int, cache: dict | None = None
    ) -> StorageRecord: ...


def storage_bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


def record_bytes(record: StorageRecord, nbytes: int) -> torch.Tensor:
    return storage_bytes(record.storage)[
        record.byte_offset : record.byte_offset + nbytes
    ]


def rebuild_entries(
    entries: Iterable[StorageEntry], device_index: int
) -> list[StorageRecord]:
    records = []
    cache: dict[object, StorageRecord] = {}
    try:
        for entry in entries:
            records.append(entry.rebuild(device_index, cache))
    except BaseException:
        records.clear()
        for base in cache.values():
            close = getattr(base.owner, "close", None)
            if close is not None:
                with suppress(BaseException):
                    close()
        raise
    return records


class IPCBackendBase(ABC):
    """Materialize shared storage and export rebuildable entries."""

    @abstractmethod
    def materialize(self, storages: list[torch.UntypedStorage]) -> list[StorageRecord]:
        """Return records in manifest order. Copied inputs may be replaced in place."""

    @abstractmethod
    def export_storage(self, record: StorageRecord) -> StorageEntry:
        """Describe a storage without copying it for each consumer."""
