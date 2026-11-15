# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared object stores that deliver auxiliary outputs to other processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from threading import Lock
from typing import Any


class AuxObjectStoreError(RuntimeError):
    """A shared auxiliary-output object could not be written."""


class AuxObjectStore(ABC):
    """Immutable named objects shared by engines and their consumers.

    A key is written once and never changes. Writing a key that already
    exists succeeds and keeps the stored bytes, so retried writes are safe.
    Objects stay until they are removed by key.
    """

    @abstractmethod
    def put_batch(self, keys: Sequence[str], values: Sequence[bytes]) -> None:
        """Write every object, or raise `AuxObjectStoreError`."""

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Return the object, or `None` if it does not exist."""

    @abstractmethod
    def exists(self, keys: Sequence[str]) -> list[bool]:
        """Return whether each object exists."""

    @abstractmethod
    def remove(self, keys: Sequence[str]) -> None:
        """Remove the objects that exist."""

    @abstractmethod
    def remove_prefix(self, prefix: str) -> int:
        """Remove every object whose key starts with `prefix`; return how many."""

    def close(self) -> None:
        """Release the client's resources."""
        return None


class LocalAuxObjectStore(AuxObjectStore):
    """Objects held in this process."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._lock = Lock()

    def put_batch(self, keys: Sequence[str], values: Sequence[bytes]) -> None:
        if len(keys) != len(values):
            raise ValueError("key and value counts differ")
        with self._lock:
            for key, value in zip(keys, values):
                self._objects.setdefault(key, bytes(value))

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._objects.get(key)

    def exists(self, keys: Sequence[str]) -> list[bool]:
        with self._lock:
            return [key in self._objects for key in keys]

    def remove(self, keys: Sequence[str]) -> None:
        with self._lock:
            for key in keys:
                self._objects.pop(key, None)

    def remove_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [key for key in self._objects if key.startswith(prefix)]
            for key in keys:
                del self._objects[key]
            return len(keys)


def create_aux_object_store(config: dict[str, Any]) -> AuxObjectStore:
    """Create the store named by `config["backend"]`."""
    if config["backend"] == "mooncake":
        from vllm.distributed.aux_output_connector.mooncake import (
            MooncakeAuxObjectStore,
        )

        return MooncakeAuxObjectStore.create()
    raise ValueError(f"Unknown auxiliary-output store backend {config['backend']!r}")
