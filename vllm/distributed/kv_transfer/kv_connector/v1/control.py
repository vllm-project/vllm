# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-free connector work and rank-specific completion accounting."""

from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_transfer import KVCacheTransferLease
from vllm.v1.kv_hints.transfers import KVTransferCompletion


@dataclass(frozen=True)
class KVTransferRankResult:
    transfer_id: int
    generation: int
    rank: int
    success: bool
    reason: str = ""
    missing: bool = False
    bytes_transferred: int = 0


@dataclass
class _PendingTransfer:
    lease: KVCacheTransferLease
    ranks: dict[int, KVTransferRankResult] = field(default_factory=dict)


class KVTransferQueue:
    def __init__(self, num_workers: int):
        if num_workers < 1:
            raise ValueError("Transfers require at least one worker")
        self.num_workers = num_workers
        self._queued: list[KVCacheTransferLease] = []
        self._pending: dict[int, _PendingTransfer] = {}
        self._completed: list[KVTransferCompletion] = []

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def submit(self, lease: KVCacheTransferLease) -> None:
        if lease.transfer_id in self._pending:
            raise ValueError("Physical transfer is already queued")
        self._pending[lease.transfer_id] = _PendingTransfer(lease)
        self._queued.append(lease)

    def take_metadata(self) -> list[KVCacheTransferLease]:
        queued, self._queued = self._queued, []
        return queued

    def update(self, results: list[KVTransferRankResult]) -> None:
        for result in results:
            pending = self._pending.get(result.transfer_id)
            if (
                pending is None
                or pending.lease.generation != result.generation
                or not 0 <= result.rank < self.num_workers
            ):
                continue
            previous = pending.ranks.get(result.rank)
            if previous is None or not result.success:
                pending.ranks[result.rank] = result
            if len(pending.ranks) != self.num_workers:
                continue
            failures = [r for r in pending.ranks.values() if not r.success]
            self._completed.append(
                KVTransferCompletion(
                    result.transfer_id,
                    result.generation,
                    not failures,
                    "; ".join(r.reason[:512] for r in failures[:3]),
                    bool(failures) and all(r.missing for r in failures),
                    sum(r.bytes_transferred for r in pending.ranks.values()),
                )
            )
            del self._pending[result.transfer_id]

    def poll(self) -> tuple[KVTransferCompletion, ...]:
        completed, self._completed = tuple(self._completed), []
        return completed
