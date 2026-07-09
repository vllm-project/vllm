"""Stubs for torch.distributed classes unavailable on ROCm Windows builds.

On Windows, torch.distributed.is_available() returns False, so classes like
PrefixStore, Store, TCPStore are not defined. This module provides minimal
stubs so that vLLM can import and run (single-GPU or without distributed
functionality) on ROCm Windows.
"""

import os
import tempfile
from typing import Any


class Store:
    """Minimal in-memory Store stub for ROCm Windows."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def set(self, key: str, value: bytes) -> None:
        self._store[key] = value

    def get(self, key: str) -> bytes:
        return self._store[key]

    def wait(self, keys: list[str], timeout: int = ...) -> None:
        pass

    def compare_set(
        self, key: str, expected_value: bytes, desired_value: bytes
    ) -> bytes:
        old = self._store.get(key, b"")
        self._store[key] = desired_value
        return old

    def delete_key(self, key: str) -> None:
        self._store.pop(key, None)


class PrefixStore:
    """Minimal PrefixStore stub that delegates to a backing Store."""

    def __init__(self, prefix: str, store: Store) -> None:
        self._prefix = prefix
        self._store = store

    def set(self, key: str, value: bytes) -> None:
        self._store.set(self._prefix + key, value)

    def get(self, key: str) -> bytes:
        return self._store.get(self._prefix + key)

    def wait(self, keys: list[str], timeout: int = 3600000) -> None:
        self._store.wait([self._prefix + k for k in keys], timeout)

    def compare_set(
        self, key: str, expected_value: bytes, desired_value: bytes
    ) -> bytes:
        return self._store.compare_set(
            self._prefix + key, expected_value, desired_value
        )

    def delete_key(self, key: str) -> None:
        self._store.delete_key(self._prefix + key)


def _ensure_dist_stubs() -> None:
    """Inject distributed stubs into torch.distributed if missing."""
    import torch.distributed as dist

    if not hasattr(dist, "PrefixStore"):
        dist.PrefixStore = PrefixStore
    if not hasattr(dist, "Store"):
        dist.Store = Store


def is_nccl_available() -> bool:
    """Stub for torch.distributed.distributed_c10d.is_nccl_available.

    On Windows ROCm, NCCL is not available, so this returns False.
    """
    return False
