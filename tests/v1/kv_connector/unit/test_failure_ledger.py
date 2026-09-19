# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Mooncake failure ledger.

The ledger is the single channel between worker-side failure origins
and scheduler-side policy: report from any thread, drain from the
scheduler, and auto-report in-flight requests past their deadline.
"""

import threading
import time

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.failure_ledger import (
    FailureLedger,
    FailureStage,
)


def test_failure_stage_taxonomy():
    """Pin the failure origins the spine recognizes."""
    assert {stage.name for stage in FailureStage} == {
        "BOOTSTRAP",
        "REGISTRATION",
        "SEND",
        "RECV",
        "STORE_PUT",
        "STORE_GET",
        "WATCHDOG",
    }


def test_report_and_drain():
    ledger = FailureLedger()
    ledger.report_failure("r1", FailureStage.SEND, "xfer error")
    ledger.report_failure("r2", FailureStage.BOOTSTRAP)
    # A second origin reporting the same request is preserved; the
    # consumer deduplicates by request_id.
    ledger.report_failure("r1", FailureStage.RECV, "late origin")

    records = ledger.take_failures()
    assert [(r.request_id, r.stage, r.detail) for r in records] == [
        ("r1", FailureStage.SEND, "xfer error"),
        ("r2", FailureStage.BOOTSTRAP, ""),
        ("r1", FailureStage.RECV, "late origin"),
    ]
    assert ledger.take_failures() == []


def test_watchdog_reports_expired_in_flight():
    ledger = FailureLedger()
    ledger.track_request("r1", timeout_s=0.05)
    ledger.track_request("r2", timeout_s=None)

    time.sleep(0.1)
    records = ledger.take_failures()
    assert [r.request_id for r in records] == ["r1"]
    assert records[0].stage == FailureStage.WATCHDOG
    # The expired request must not resurface on the next drain.
    assert ledger.take_failures() == []


@pytest.mark.parametrize("terminal_state", ["release", "report"])
def test_terminal_state_prevents_watchdog(terminal_state):
    """In-flight tracking ends only via a terminal state, and a
    terminal state never resurfaces as a watchdog failure."""
    ledger = FailureLedger()
    ledger.track_request("r1", timeout_s=0.05)
    if terminal_state == "release":
        ledger.release_request("r1")
    else:
        ledger.report_failure("r1", FailureStage.RECV)

    time.sleep(0.1)
    records = ledger.take_failures()
    if terminal_state == "report":
        # Compare fields; reported_at is set at report time.
        assert [(r.request_id, r.stage) for r in records] == [
            ("r1", FailureStage.RECV),
        ]
    else:
        assert records == []
    assert ledger.take_failures() == []


def test_concurrent_report_from_threads():
    ledger = FailureLedger()
    num_threads, per_thread = 8, 250

    def worker(base: int):
        for i in range(per_thread):
            ledger.report_failure(f"r{base + i}", FailureStage.SEND)

    threads = [
        threading.Thread(target=worker, args=(t * per_thread,))
        for t in range(num_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = ledger.take_failures()
    assert len(records) == num_threads * per_thread
    assert len({r.request_id for r in records}) == num_threads * per_thread


def test_concurrent_report_and_drain():
    """Drains and reports may interleave; no failure may be lost or duplicated."""
    ledger = FailureLedger()
    total = 1000
    drained: list[str] = []
    lock = threading.Lock()

    def reporter():
        for i in range(total):
            ledger.report_failure(f"r{i}", FailureStage.SEND)

    def drainer():
        while True:
            records = ledger.take_failures()
            with lock:
                drained.extend(r.request_id for r in records)
                if len(drained) >= total:
                    return

    producer = threading.Thread(target=reporter)
    # Daemon: a stuck consumer must not block the pytest process at exit.
    consumer = threading.Thread(target=drainer, daemon=True)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join(timeout=30)

    assert sorted(drained) == sorted(f"r{i}" for i in range(total))
