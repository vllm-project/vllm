# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Failure ledger for the Mooncake connectors.

The Mooncake connectors span several async worlds (scheduler loop,
worker threads, send/receive loops, bootstrap server). Failures
originate in the async worlds while the fail-or-recompute decision
belongs to the scheduler. This module is the single channel between
the two sides:

- failure origins report through `FailureLedger.report_failure` from
  any thread;
- the connector drains reported failures each engine step (where
  transfers complete) and forwards the affected requests to the
  scheduler, which applies `kv_load_failure_policy`;
- in-flight requests are auto-reported by the same drain once their
  deadline passes, so a missing report degrades to a clean timeout
  instead of a stranded request. Deadlines are checked only while
  the engine steps; the ledger runs no timer of its own.

The ledger only records and reports; policy stays in the scheduler.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field


class FailureStage(enum.Enum):
    """Where in the transfer pipeline a failure originated.

    `WATCHDOG` marks a request whose deadline passed without any
    origin reporting; the ledger itself is the origin.
    """

    BOOTSTRAP = enum.auto()
    REGISTRATION = enum.auto()
    SEND = enum.auto()
    RECV = enum.auto()
    STORE_PUT = enum.auto()
    STORE_GET = enum.auto()
    WATCHDOG = enum.auto()


@dataclass(frozen=True)
class FailureRecord:
    """One reported failure, ready for scheduler-side policy."""

    request_id: str
    stage: FailureStage
    detail: str = ""
    # Clock used by the transfer deadlines (seconds).
    reported_at: float = field(default_factory=time.perf_counter)


class FailureLedger:
    """Thread-safe ledger of transfer failures for one connector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures: list[FailureRecord] = []
        # request_id -> monotonic deadline (seconds); None = no watchdog.
        self._in_flight: dict[str, float | None] = {}

    def track_request(self, request_id: str, timeout_s: float | None = None) -> None:
        """Mark a request as in flight, with an optional watchdog deadline.

        Re-tracking a request replaces its previous deadline.
        """
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s
        with self._lock:
            self._in_flight[request_id] = deadline

    def release_request(self, request_id: str) -> None:
        """Drop in-flight tracking after a terminal state (success or failure)."""
        with self._lock:
            self._in_flight.pop(request_id, None)

    def report_failure(
        self, request_id: str, stage: FailureStage, detail: str = ""
    ) -> None:
        """Report one failure from any thread; clears in-flight tracking.

        Multiple origins may report the same request; consumers
        deduplicate by `request_id`.
        """
        with self._lock:
            self._in_flight.pop(request_id, None)
            self._failures.append(
                FailureRecord(request_id=request_id, stage=stage, detail=detail)
            )

    def take_failures(self) -> list[FailureRecord]:
        """Drain reported failures (scheduler side).

        In-flight requests past their deadline are auto-reported with
        `FailureStage.WATCHDOG` and removed from tracking first, so a
        missing report degrades to a clean timeout.
        """
        now = time.perf_counter()
        with self._lock:
            expired = [
                request_id
                for request_id, deadline in self._in_flight.items()
                if deadline is not None and deadline <= now
            ]
            for request_id in expired:
                del self._in_flight[request_id]
                self._failures.append(
                    FailureRecord(
                        request_id=request_id,
                        stage=FailureStage.WATCHDOG,
                        detail="in-flight deadline exceeded",
                    )
                )
            drained, self._failures = self._failures, []
        return drained
