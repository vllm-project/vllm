# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async receive admission, terminal reporting and target quarantine ownership."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults

from .data import CopyJob, PeerWorkerMetadata
from .session import Grant


@dataclass
class AsyncRead:
    job: CopyJob
    deadline: float
    timeout_s: float
    submitted: bool = False
    terminal: bool = False
    failed: bool = False
    drained: bool = False
    reported: bool = False
    quarantined: bool = False
    released: bool = False
    failure_reason: str | None = None
    resources: list[Any] = field(default_factory=list)
    drain_query: Callable[[], bool] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def identity(self) -> str:
        grant = self.job.grant
        return f"{grant.endpoint}/{grant.epoch}/{grant.reader}/{grant.sequence}"

    def begin(self) -> bool:
        """Close the timeout-versus-submit race before a possible device post."""
        with self._lock:
            if self.terminal:
                return False
            now = monotonic()
            if now >= self.deadline:
                self._fail_deadline()
                self.drained = not self.submitted
                return False
            if not self.submitted:
                self.deadline = min(self.deadline, now + self.timeout_s)
            self.submitted = True
            return True

    def complete(
        self,
        success: bool,
        *,
        drained: bool,
        drain_query: Callable[[], bool] | None = None,
        reason: str = "exception",
    ) -> None:
        with self._lock:
            if not self.failed:
                if monotonic() >= self.deadline:
                    self._fail_deadline()
                elif not success:
                    self.failed = True
                    self.failure_reason = reason
            self.terminal = True
            self.drained |= drained or not self.submitted
            self.drain_query = drain_query

    def _fail_deadline(self) -> None:
        self.terminal = self.failed = True
        self.failure_reason = (
            "deadline_after_post" if self.submitted else "deadline_before_post"
        )

    def cancel_unsubmitted(self, reason: str = "cancelled") -> None:
        with self._lock:
            if not self.submitted and not self.terminal:
                self.terminal = self.failed = self.drained = True
                self.failure_reason = reason

    def snapshot(self) -> tuple[bool, bool, bool]:
        with self._lock:
            if not self.terminal and monotonic() >= self.deadline:
                self._fail_deadline()
                self.drained = not self.submitted
            if self.terminal and not self.drained and self.drain_query is not None:
                try:
                    self.drained = self.drain_query()
                except Exception:
                    self.drained = False
            return self.terminal, self.failed, self.drained


class AsyncReceives:
    def __init__(self, release: Callable[[Grant], None]):
        self.release = release
        self.reads: dict[str, AsyncRead] = {}
        self.quarantined: dict[str, tuple[int, ...]] = {}
        self.drained: set[str] = set()

    def add(self, job: CopyJob, timeout: float) -> AsyncRead:
        grant = job.grant
        if grant.lease_s is None or not job.request_id:
            raise ValueError("Async receives require a request ID and leased grant")
        # Queue/setup/step-fence time uses the grant budget. Only the first
        # possible post starts the transfer timeout, without extending the lease.
        read = AsyncRead(job, grant.acquired_before + grant.lease_s / 2, timeout)
        if read.identity in self.reads:
            raise ValueError("Duplicate async receive")
        if any(
            r.job.request_id == job.request_id and not r.reported
            for r in self.reads.values()
        ):
            raise ValueError("A request may have only one active receive")
        self.reads[read.identity] = read
        return read

    def poll(
        self, finished: set[str] | frozenset[str] = frozenset()
    ) -> KVConnectorTransferResults:
        result = KVConnectorTransferResults()
        for identity, read in list(self.reads.items()):
            if read.job.request_id in finished:
                read.cancel_unsubmitted()
            terminal, failed, drained = read.snapshot()
            if not terminal:
                continue
            if not read.reported:
                result.finished_recving.add(read.job.request_id)
                if failed:
                    result.failed_recving.add(read.job.request_id)
                if not drained:
                    self.quarantined[identity] = tuple(
                        sorted({target for _, _, target in read.job.blocks})
                    )
                    read.quarantined = True
                read.reported = True
            if drained:
                if not read.released:
                    self.release(read.job.grant)
                    read.released = True
                if read.quarantined:
                    self.drained.add(identity)
                del self.reads[identity]
        return result

    def metadata(self, step: int) -> PeerWorkerMetadata:
        result = PeerWorkerMetadata(step, self.quarantined, self.drained)
        self.quarantined, self.drained = {}, set()
        return result
