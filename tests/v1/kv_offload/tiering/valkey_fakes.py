# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-memory Valkey clients for the valkey tier tests."""

import threading
import time
from collections import defaultdict, deque

from vllm.v1.kv_offload.tiering.valkey_large_object.client import (
    ValkeyTransportError,
)


class FakeValkeyClient:
    """In-memory ValkeyClient. Exceptions queued in ``failures[operation]``
    are raised by the next calls to that operation.

    Clients built over the same ``values`` see the same server data, as a
    reconnected client would.
    """

    def __init__(self, values: dict[str, bytes] | None = None) -> None:
        self.values: dict[str, bytes] = {} if values is None else values
        self.ttls: dict[str, int | None] = {}
        self.failures: defaultdict[str, deque[Exception]] = defaultdict(deque)
        self.closed = False

    def _raise_queued_failure(self, operation: str) -> None:
        if self.failures[operation]:
            raise self.failures[operation].popleft()

    def set(self, key: str, value: memoryview, ttl_seconds: int | None) -> None:
        self._raise_queued_failure("set")
        self.values[key] = bytes(value)
        self.ttls[key] = ttl_seconds

    def get_into(self, key: str, buffer: memoryview) -> int | None:
        self._raise_queued_failure("get")
        value = self.values.get(key)
        if value is None:
            return None
        if len(buffer) < len(value):
            raise ValueError("value larger than buffer")
        buffer[: len(value)] = value
        return len(value)

    def exists_each(self, keys: list[str]) -> list[bool]:
        return [key in self.values for key in keys]

    def close(self) -> None:
        self.closed = True


class FakeLargeObjectClient(FakeValkeyClient):
    """In-memory LargeObjectClient.

    As in glide, a region serves one transfer at a time: a transfer on a busy
    region records the violation in ``overlapping_transfers`` and fails.
    ``transfer_seconds`` stretches transfers so overlaps would show. With
    ``hang_transfers`` set, transfers block until ``close()`` cancels them.
    ``events`` records region lifecycle and cancellation order.
    """

    def __init__(self, values: dict[str, bytes] | None = None) -> None:
        super().__init__(values)
        self.regions: list[memoryview] = []
        self.hang_transfers = False
        self.transfer_seconds = 0.0
        self.overlapping_transfers = 0
        self.events: list[str] = []
        self._busy: set[int] = set()
        self._busy_lock = threading.Lock()
        self._closed = threading.Event()

    def register_region(self, memory: memoryview) -> int:
        self.regions.append(memory.cast("B"))
        self.events.append("register_region")
        return len(self.regions) - 1

    def _block_until_closed(self) -> None:
        self._closed.wait()
        self.events.append("transfer_cancelled")
        raise ValkeyTransportError("client closed")

    def _take(self, region: int) -> memoryview:
        with self._busy_lock:
            if region in self._busy:
                self.overlapping_transfers += 1
                raise RuntimeError(f"region {region} is serving another transfer")
            self._busy.add(region)
        time.sleep(self.transfer_seconds)
        return self.regions[region]

    def _give_back(self, region: int) -> None:
        with self._busy_lock:
            self._busy.discard(region)

    def rdma_set(self, key: str, region: int, offset: int, length: int) -> None:
        self._raise_queued_failure("rdma_set")
        if self.hang_transfers:
            self._block_until_closed()
        memory = self._take(region)
        try:
            self.values[key] = bytes(memory[offset : offset + length])
        finally:
            self._give_back(region)

    def rdma_get(self, key: str, region: int, offset: int, length: int) -> int | None:
        self._raise_queued_failure("rdma_get")
        if self.hang_transfers:
            self._block_until_closed()
        memory = self._take(region)
        try:
            value = self.values.get(key)
            if value is None:
                return None
            if length < len(value):
                raise ValueError("value larger than window")
            memory[offset : offset + len(value)] = value
            return len(value)
        finally:
            self._give_back(region)

    def expire(self, key: str, ttl_seconds: int) -> None:
        self.ttls[key] = ttl_seconds

    def close(self) -> None:
        super().close()
        self.events.append("close")
        self._closed.set()

    def release_regions(self) -> None:
        self.events.append("release_regions")
