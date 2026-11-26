# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded lookups for queued requests; never retain embedding tensors."""

import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from .data import TensorSpec
from .store_client import EmbeddingStoreOperationError


@dataclass
class _Lookup:
    expected: dict[str, TensorSpec]
    future: Future[tuple[dict[str, bool], float]]


class StoreLookupPrefetcher:
    def __init__(
        self,
        lookup: Callable[[Mapping[str, TensorSpec]], dict[str, bool]],
        *,
        max_items: int,
        ttl_s: float,
    ) -> None:
        self._lookup = lookup
        self._max_items = max_items
        self._ttl_s = ttl_s
        self._entries: dict[str, _Lookup] = {}
        self._retired: list[_Lookup] = []
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ec-store-lookup"
        )

    def _run(self, expected: Mapping[str, TensorSpec]) -> tuple[dict[str, bool], float]:
        return self._lookup(expected), time.monotonic()

    def submit(self, tag: str, expected: Mapping[str, TensorSpec]) -> bool:
        self._prune()
        if tag in self._entries:
            return True
        count = sum(len(entry.expected) for entry in self._entries.values())
        count += sum(len(entry.expected) for entry in self._retired)
        if not expected or count + len(expected) > self._max_items:
            return False
        snapshot = dict(expected)
        self._entries[tag] = _Lookup(
            snapshot, self._executor.submit(self._run, snapshot)
        )
        return True

    def take(
        self, tag: str, expected: Mapping[str, TensorSpec]
    ) -> dict[str, bool] | None:
        """At actual use, reuse the lookup or let the caller query normally."""
        entry = self._entries.pop(tag, None)
        if entry is None:
            return None
        try:
            # No extra scheduling gate: wait only where normal lookup would run.
            result, completed_at = entry.future.result()
        except (EmbeddingStoreOperationError, OSError):
            return None
        if time.monotonic() - completed_at > self._ttl_s:
            return None
        return {
            key: result[key]
            for key, spec in expected.items()
            if entry.expected.get(key) == spec and key in result
        }

    def cancel(self, tag: str) -> None:
        entry = self._entries.pop(tag, None)
        if entry is not None:
            entry.future.cancel()
            if not entry.future.done():
                self._retired.append(entry)

    def _prune(self) -> None:
        self._retired = [entry for entry in self._retired if not entry.future.done()]
        now = time.monotonic()
        for tag, entry in tuple(self._entries.items()):
            if entry.future.done():
                try:
                    _, completed_at = entry.future.result()
                except Exception:
                    self.cancel(tag)
                    continue
                if now - completed_at > self._ttl_s:
                    self.cancel(tag)

    def reset(self) -> None:
        for tag in tuple(self._entries):
            self.cancel(tag)

    def shutdown(self) -> None:
        self.reset()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._retired.clear()
