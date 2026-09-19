# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the NIXL checkpoint lifecycle API."""

import contextlib
import queue
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1 import multi_connector
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    base_scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlHandshakePayload,
)


class _TestConnector(NixlBaseConnector):
    def start_load_kv(self, *args, **kwargs):
        raise NotImplementedError


def _publishing_worker(
    tp_rank: int = 0,
    world_size: int = 1,
    pcp_rank: int = 0,
    pcp_dcp_sharded: bool = False,
    kv_role: str = "kv_both",
) -> NixlBaseConnectorWorker:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.engine_id = "engine-1"
    worker.pp_rank = 0
    worker.tp_rank = tp_rank
    worker.world_size = world_size
    worker.pcp_rank = pcp_rank
    worker.pcp_dcp_sharded = pcp_dcp_sharded
    worker.kv_transfer_config = SimpleNamespace(kv_role=kv_role)
    worker.xfer_handshake_metadata = NixlHandshakePayload(
        compatibility_hash="hash", agent_metadata_bytes=b"metadata"
    )
    return worker


def test_release_for_checkpoint_drops_nixl_wrapper() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.nixl_wrapper = object()

    worker.release_for_checkpoint()

    assert worker.nixl_wrapper is None


def test_connector_forwards_checkpoint_lifecycle() -> None:
    worker = SimpleNamespace(
        quiesce=Mock(),
        release_for_checkpoint=Mock(),
    )
    connector = object.__new__(_TestConnector)
    connector.connector_worker = worker

    connector.quiesce(timeout=30.0)
    connector.release_for_checkpoint()

    worker.quiesce.assert_called_once_with(30.0)
    worker.release_for_checkpoint.assert_called_once_with()


def test_multi_connector_forwards_checkpoint_release() -> None:
    first = SimpleNamespace(release_for_checkpoint=Mock())
    second = SimpleNamespace(release_for_checkpoint=Mock())
    connector = object.__new__(multi_connector.MultiConnector)
    connector._connectors = [first, second]

    connector.release_for_checkpoint()

    first.release_for_checkpoint.assert_called_once_with()
    second.release_for_checkpoint.assert_called_once_with()


def test_scheduler_refreshes_after_ip_change(monkeypatch) -> None:
    scheduler = object.__new__(NixlBaseConnectorScheduler)
    scheduler.side_channel_host = "10.0.0.1"
    scheduler._nixl_handshake_listener_t = None
    scheduler._encoded_handshake_data = {}
    monkeypatch.setattr(socket, "gethostname", lambda: "pod")
    monkeypatch.setattr(socket, "gethostbyname", lambda _: "10.0.0.2")

    scheduler.refresh_handshake_endpoint()

    assert scheduler.side_channel_host == "10.0.0.2"


def test_worker_reinitialization_refreshes_scheduler_first(monkeypatch) -> None:
    events = []
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.engine_id = "engine"
    worker.pp_rank = 0
    worker.tp_rank = 0
    worker.nixl_wrapper = None
    worker._registered_kv_caches = {"layer": object()}
    worker._nixl_config = object()
    worker._nixl_wrapper_cls = lambda *_: events.append("new-agent") or object()
    worker._create_handshake_executor = lambda: events.append("new-executor") or Mock()
    worker.register_kv_caches = lambda _: events.append("register-caches")
    worker._publish_handshake_metadata = lambda: events.append("publish-metadata")
    worker._refresh_local_scheduler = lambda: events.append("refresh-scheduler")

    worker.reinitialize()

    assert events == [
        "refresh-scheduler",
        "new-agent",
        "new-executor",
        "register-caches",
        "publish-metadata",
    ]


def test_refresh_local_scheduler_uses_registry(monkeypatch) -> None:
    # reinitialize() must talk to the co-located scheduler directly: this
    # milestone's scope excludes P/D and multi-GPU topologies (see
    # RFC-nixl-connector-lifecycle.md), so scheduler and worker always share
    # one EngineCore process and a same-process call is correct and simpler
    # than a ZMQ round trip to a listener address the worker cannot know.
    scheduler = SimpleNamespace(refresh_handshake_endpoint=Mock())
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.engine_id = "engine-1"
    monkeypatch.setattr(
        base_scheduler, "get_local_nixl_scheduler", lambda engine_id: scheduler
    )

    worker._refresh_local_scheduler()

    scheduler.refresh_handshake_endpoint.assert_called_once_with()


def test_refresh_local_scheduler_requires_local_scheduler(
    monkeypatch,
) -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.engine_id = "engine-1"
    monkeypatch.setattr(
        base_scheduler, "get_local_nixl_scheduler", lambda engine_id: None
    )

    with pytest.raises(RuntimeError, match="local EngineCore process"):
        worker._refresh_local_scheduler()


def test_quiesce_drains_all_lifecycle_work_before_stopping_threads() -> None:
    events = []
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._recving_metadata = {"request": object()}
    worker._quiesce_drained_sending = set()
    worker._quiesce_drained_recving = set()
    worker._quiesce_drained_failed = set()

    def get_transfer_results():
        events.append("get-transfer-results")
        worker._recving_metadata.clear()
        return KVConnectorTransferResults()

    worker.get_transfer_results = get_transfer_results
    worker._stop_push_writer = lambda: events.append("stop-push-writer")
    worker._stop_handshake_executor = lambda: events.append("stop-handshake-executor")
    worker._discard_push_work = lambda: events.append("discard-push-work")
    worker._release_transport_state = lambda: events.append("release-transport")

    worker.quiesce(timeout=1.0)

    assert events == [
        "get-transfer-results",
        "stop-push-writer",
        "stop-handshake-executor",
        "discard-push-work",
        "release-transport",
    ]


def test_quiesce_tracks_handshake_work() -> None:
    # Push-mode's own queues are checked by NixlPushConnectorWorker's
    # override (tests/v1/kv_connector/unit/test_nixl_push_connector.py);
    # the base class only knows about its own (pull-mode) state.
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._handshake_futures = {"engine": object()}
    worker._ready_requests = object()

    assert worker._pending_lifecycle_work() == (
        "_handshake_futures",
        "_ready_requests",
    )


def test_quiesce_tracks_request_failure_and_notification_work() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._pending_recv_notifs = {"request": [("agent", b"notif")]}
    worker._failed_inflight_recvs = {"request"}
    worker._failed_recv_reqs = queue.Queue()
    worker._failed_recv_reqs.put("request")
    worker._replicated_pcp_done_sending = {"request"}

    assert set(worker._pending_lifecycle_work()) == {
        "_pending_recv_notifs",
        "_failed_inflight_recvs",
        "_failed_recv_reqs",
        "_replicated_pcp_done_sending",
    }


def test_quiesce_accumulates_ids_drained_while_stopping() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._recving_metadata = {"request": object()}
    worker._quiesce_drained_sending = set()
    worker._quiesce_drained_recving = set()
    worker._quiesce_drained_failed = set()

    def get_transfer_results():
        worker._recving_metadata.clear()
        return KVConnectorTransferResults(
            finished_sending={"sent-1"},
            finished_recving={"recv-1", "failed-1"},
            failed_recving={"failed-1"},
        )

    worker.get_transfer_results = get_transfer_results
    worker._stop_push_writer = lambda: None
    worker._stop_handshake_executor = lambda: None
    worker._discard_push_work = lambda: None
    worker._release_transport_state = lambda: None

    worker.quiesce(timeout=1.0)

    assert worker._quiesce_drained_sending == {"sent-1"}
    assert worker._quiesce_drained_recving == {"recv-1", "failed-1"}
    assert worker._quiesce_drained_failed == {"failed-1"}


def test_transfer_results_report_ids_drained_during_quiesce() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.transfer_topo = object()
    worker.tp_rank = 0
    worker.use_host_buffer = False
    worker._recving_transfers = {}
    worker._recving_metadata = {}
    worker._failed_recv_reqs = queue.Queue()
    worker._recv_failures = set()
    worker._replicated_pcp_done_sending = set()
    worker._reqs_to_send = {}
    worker._get_new_notifs = lambda: set()
    worker._pop_done_transfers = lambda _transfers: (set(), set())
    worker._quiesce_drained_sending = {"sent-1"}
    worker._quiesce_drained_recving = {"recv-1", "failed-1"}
    worker._quiesce_drained_failed = {"failed-1"}

    results = worker.get_transfer_results()

    assert results.finished_sending == {"sent-1"}
    assert results.finished_recving == {"recv-1", "failed-1"}
    # A failure drained by quiesce must still reach the scheduler as a failure.
    assert results.failed_recving == {"failed-1"}
    # Reported exactly once: the accumulator is cleared after being drained.
    assert worker._quiesce_drained_sending == set()
    assert worker._quiesce_drained_recving == set()
    assert worker._quiesce_drained_failed == set()


def test_quiesce_timeout_resets_quiescing_flag() -> None:
    # No thread/executor teardown happened, so it is safe to resume normal
    # handshake operation after a drain timeout.
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._pending_lifecycle_work = lambda: ("_ready_requests",)
    worker.get_transfer_results = KVConnectorTransferResults
    worker._quiesce_drained_sending = set()
    worker._quiesce_drained_recving = set()
    worker._quiesce_drained_failed = set()

    with contextlib.suppress(TimeoutError):
        worker.quiesce(timeout=0.0)

    assert worker._checkpoint_quiescing is False


def test_quiesce_keeps_quiescing_flag_after_teardown_before_abort() -> None:
    # Once the handshake executor and push writer are stopped, only
    # reinitialize() can restore them; the flag must stay set so callers
    # keep hitting the guarded "quiescing" error instead of using torn-down
    # state (e.g. a None executor).
    worker = object.__new__(NixlBaseConnectorWorker)
    pending_calls = [(), ("_ready_requests",)]
    worker._pending_lifecycle_work = lambda: pending_calls.pop(0)
    worker.get_transfer_results = KVConnectorTransferResults
    worker._quiesce_drained_sending = set()
    worker._quiesce_drained_recving = set()
    worker._quiesce_drained_failed = set()
    worker._stop_push_writer = lambda: None
    worker._stop_handshake_executor = lambda: None

    with contextlib.suppress(RuntimeError):
        worker.quiesce(timeout=1.0)

    assert worker._checkpoint_quiescing is True


def test_publish_handshake_metadata_uses_local_scheduler(monkeypatch) -> None:
    # Same reasoning as _refresh_local_scheduler: this milestone's
    # scope excludes P/D and multi-GPU topologies, so scheduler and worker
    # always share one EngineCore process and a same-process call is
    # correct -- a ZMQ round trip has no reliable address to dial (see
    # RFC-nixl-connector-lifecycle.md).
    scheduler = SimpleNamespace(update_handshake_metadata=Mock())
    worker = _publishing_worker()
    monkeypatch.setattr(
        base_scheduler, "get_local_nixl_scheduler", lambda engine_id: scheduler
    )

    worker._publish_handshake_metadata()

    scheduler.update_handshake_metadata.assert_called_once_with(
        0, 0, worker.xfer_handshake_metadata
    )


def test_publish_handshake_metadata_keys_sharded_pcp_by_transfer_rank(
    monkeypatch,
) -> None:
    # Must match the (pp, tp + pcp * tp_size) key the GPU worker serves,
    # or every PCP rank overwrites the same payload.
    scheduler = SimpleNamespace(update_handshake_metadata=Mock())
    worker = _publishing_worker(
        tp_rank=1, world_size=2, pcp_rank=1, pcp_dcp_sharded=True
    )
    monkeypatch.setattr(
        base_scheduler, "get_local_nixl_scheduler", lambda engine_id: scheduler
    )

    worker._publish_handshake_metadata()

    scheduler.update_handshake_metadata.assert_called_once_with(
        0, 3, worker.xfer_handshake_metadata
    )


def test_publish_handshake_metadata_skips_replicated_pcp_producer_rank(
    monkeypatch,
) -> None:
    # Replicated PCP producer ranks > 0 never transfer and serve no metadata.
    worker = _publishing_worker(pcp_rank=1, kv_role="kv_producer")
    monkeypatch.setattr(
        base_scheduler,
        "get_local_nixl_scheduler",
        lambda engine_id: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    worker._publish_handshake_metadata()


def test_publish_handshake_metadata_requires_local_scheduler(monkeypatch) -> None:
    worker = _publishing_worker()
    monkeypatch.setattr(
        base_scheduler, "get_local_nixl_scheduler", lambda engine_id: None
    )

    with pytest.raises(RuntimeError, match="local EngineCore process"):
        worker._publish_handshake_metadata()


def test_publish_handshake_metadata_skips_without_metadata(monkeypatch) -> None:
    """No agent metadata yet -> nothing to publish, no scheduler lookup."""
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.xfer_handshake_metadata = None
    monkeypatch.setattr(
        base_scheduler,
        "get_local_nixl_scheduler",
        lambda engine_id: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    worker._publish_handshake_metadata()
