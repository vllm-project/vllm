# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


class DummyConnector:
    def __init__(self) -> None:
        self.outputs: list[KVConnectorOutput] = []
        self.pending_deferred_sends = False

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        self.outputs.append(output)

    def has_pending_deferred_sends(self) -> bool:
        return self.pending_deferred_sends


def make_scheduler(requests: dict[str, object] | None = None) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.connector = DummyConnector()
    scheduler.requests = requests or {}
    scheduler.finished_req_ids = set()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.running = []
    return scheduler


def test_late_finished_sending_for_removed_request_is_ignored() -> None:
    scheduler = make_scheduler()

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_sending={"already-finished"})
    )

    assert scheduler.requests == {}
    assert len(scheduler.connector.outputs) == 1


def test_late_finished_recving_for_removed_request_is_ignored() -> None:
    scheduler = make_scheduler()

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={"already-aborted"})
    )

    assert scheduler.finished_recving_kv_req_ids == set()
    assert len(scheduler.connector.outputs) == 1


def test_live_finished_sending_still_frees_blocks() -> None:
    request = SimpleNamespace(request_id="live")
    scheduler = make_scheduler({"live": request})
    freed: list[object] = []
    scheduler._free_blocks = freed.append

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_sending={"live"})
    )

    assert freed == [request]


def test_live_finished_recving_still_marks_waiting_request() -> None:
    request = SimpleNamespace(status=RequestStatus.WAITING_FOR_REMOTE_KVS)
    scheduler = make_scheduler({"live": request})

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={"live"})
    )

    assert scheduler.finished_recving_kv_req_ids == {"live"}


def test_pending_deferred_send_keeps_scheduler_active() -> None:
    scheduler = make_scheduler()
    assert not scheduler.has_finished_requests()

    scheduler.connector.pending_deferred_sends = True

    assert scheduler.has_finished_requests()
