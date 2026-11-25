# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async receive admission, terminal reporting and target quarantine ownership."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.logger import init_logger

from .data import CopyJob, PeerWorkerMetadata
from .session import Grant
from .stats import PeerL1Stats

logger = init_logger(__name__)


@dataclass
class AsyncRead:
    """Share receive state between the model thread and the RDMA I/O thread.

    `_lock` serializes submission, timeout and completion decisions. `resources`
    only keeps sessions, batches and their DMA arguments alive until drain.
    """

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
    num_bytes: int = 0
    failure_reason: str | None = None
    resources: list[Any] = field(default_factory=list)
    drain_query: Callable[[], bool] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def identity(self) -> str:
        grant = self.job.grant
        return f"{grant.endpoint}/{grant.epoch}/{grant.reader}/{grant.sequence}"

    def begin(self, num_bytes: int = 0) -> bool:
        """Mark a possible post under the lock before calling the provider.

        False forbids posting. Once true, timeout cannot claim the read was
        unsubmitted even if the provider raises without returning status.
        """
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
            self.num_bytes += num_bytes
            return True

    def complete(
        self,
        success: bool,
        *,
        drained: bool,
        drain_query: Callable[[], bool] | None = None,
        reason: str = "exception",
    ) -> None:
        """Record I/O-thread completion under the lock, preserving prior failure."""
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
        """Poll state and drain proof under the lock on the model thread."""
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
    """Report receive outcomes and quarantine ownership on the model thread.

    Receive state changes cross threads through `AsyncRead`'s locked methods.
    Source grants and target keep-alives are released only after DMA drains.
    """

    def __init__(
        self,
        release: Callable[[Grant], None],
        stats: PeerL1Stats | None = None,
        retire: Callable[[str], None] | None = None,
    ):
        self.release = release
        self.reads: dict[str, AsyncRead] = {}
        self.quarantined: dict[str, tuple[int, ...]] = {}
        self.drained: set[str] = set()
        self.failed_peers: set[str] = set()
        self.successful_peers: dict[str, float] = {}
        self.stats = stats
        self.retire = retire
        self._failure_logs: dict[tuple[str, str], float] = {}

    def add(self, job: CopyJob, timeout: float) -> AsyncRead:
        """Admit within half the original lease, including queue and setup time."""
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
        """Report terminal requests without waiting for provider or GPU completion.

        A failed, undrained read also reports target rows for scheduler-side
        quarantine before core can free request references.
        """
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
                    self.failed_peers.add(read.job.grant.endpoint)
                    reason = read.failure_reason or "exception"
                    key = read.job.grant.endpoint, reason
                    now = monotonic()
                    if now - self._failure_logs.get(key, float("-inf")) >= 60:
                        logger.info(
                            "Peer L1 RDMA receive failed owner=%s reason=%s; "
                            "recomputing request",
                            *key,
                        )
                        self._failure_logs[key] = now
                else:
                    endpoint = read.job.grant.endpoint
                    self.successful_peers[endpoint] = max(
                        self.successful_peers.get(endpoint, float("-inf")),
                        read.job.grant.acquired_before,
                    )
                if self.stats is not None:
                    self.stats.inc("worker_rdma_reads", int(read.submitted))
                    role = (
                        read.job.memory.role if read.job.memory is not None else "full"
                    )
                    self.stats.inc(
                        "worker_copy_bytes",
                        read.num_bytes,
                        transport="rdma",
                        owner_role=role,
                    )
                    if failed:
                        self.stats.inc(
                            "worker_rdma_failures", reason=reason, owner_role=role
                        )
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
                if self.retire is not None:
                    self.retire(identity)
                del self.reads[identity]
        return result

    def metadata(self, step: int) -> PeerWorkerMetadata:
        """Consume transfer-scoped quarantine and drain updates for the scheduler."""
        result = PeerWorkerMetadata(
            step,
            self.quarantined,
            self.drained,
            self.failed_peers,
            self.successful_peers,
        )
        self.quarantined, self.drained, self.failed_peers = {}, set(), set()
        self.successful_peers = {}
        return result
