# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Protocol, cast


class AdmissionCounterBuffer(Protocol):
    def __getitem__(self, index: int) -> int: ...

    def __setitem__(self, index: int, value: int) -> None: ...


class SharedAdmissionStats:
    """Lock-free admission stats shared by API server processes.

    Each process writes exclusively to its own cache-line-sized slot. Readers
    aggregate the slots, avoiding both inter-process locks and cache-line
    contention on the request path.
    """

    CACHE_LINE_BYTES = 64
    COUNTER_BYTES = 8
    COUNTERS_PER_SLOT = CACHE_LINE_BYTES // COUNTER_BYTES

    REQUEST_COUNT = 0

    def __init__(
        self,
        client_addresses: dict[str, Any] | None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        counters = None
        if client_addresses is not None:
            counters = cast(
                AdmissionCounterBuffer,
                client_addresses.pop("mp_admission_counters", None),
            )
        if counters is None:
            counters = [0] * self.num_counters(1)
            client_count = 1
            client_index = 0
        self._counters = counters
        self._client_count = client_count
        self._slot_offset = client_index * self.COUNTERS_PER_SLOT

    @classmethod
    def num_counters(cls, client_count: int) -> int:
        return client_count * cls.COUNTERS_PER_SLOT

    def _get_global(self, counter: int) -> int:
        return sum(
            self._counters[client_index * self.COUNTERS_PER_SLOT + counter]
            for client_index in range(self._client_count)
        )

    def set_num_requests(self, count: int) -> None:
        self._counters[self._slot_offset + self.REQUEST_COUNT] = count

    def get_num_requests(self) -> int:
        return self._get_global(self.REQUEST_COUNT)
