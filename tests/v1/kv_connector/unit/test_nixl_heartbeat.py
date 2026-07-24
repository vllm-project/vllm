# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the scheduler-driven heartbeat / lease-renewal system."""

import time
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HeartbeatInfo,
    NixlConnectorMetadata,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import create_request, make_nixl_scheduler

_ENGINE_A = "my-engine-id"


def _sched(kv_lease_duration: int = 30):
    return make_nixl_scheduler(heartbeat=True, kv_lease_duration=kv_lease_duration)


def _req(request_id: int = 1):
    return create_request(request_id=request_id, do_remote_prefill=True)


def _worker_stub():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w._reqs_to_send = {}
    w._lease_extension = 20
    return w


# ===================================================================
# Scheduler: on_new_request
# ===================================================================


def test_on_new_request_tracks_and_groups():
    """Add two reqs to same engine, one to another; verify grouping."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    assert s._heartbeat_by_engine[_ENGINE_A].req_ids == {"prefill-1", "prefill-2"}
    info = s._heartbeat_by_engine[_ENGINE_A]
    assert (info.host, info.port, info.tp_size) == ("my-host", 1234, 1)
    assert s._heartbeat_req_engine["id-1"] == (_ENGINE_A, "prefill-1")

    # Different engine.
    r3 = _req(3)
    r3.kv_transfer_params["remote_engine_id"] = "engine-b"
    s.on_new_request(r3)
    assert len(s._heartbeat_by_engine) == 2


def test_on_new_request_preserves_all_parallel_sizes():
    s = _sched()
    request = _req(1)
    request.kv_transfer_params.update(
        {
            "tp_size": 4,
            "dcp_size": 2,
            "pcp_size": 2,
            "pp_size": 3,
        }
    )

    s.on_new_request(request)

    info = s._heartbeat_by_engine[_ENGINE_A]
    assert (info.tp_size, info.dcp_size, info.pcp_size, info.pp_size) == (
        4,
        2,
        2,
        3,
    )


@pytest.mark.parametrize(
    "make_req",
    [
        lambda: create_request(request_id=2, do_remote_decode=True),
        lambda: create_request(request_id=3),  # no kv_transfer_params
    ],
    ids=["decode", "plain"],
)
def test_on_new_request_ignores_non_prefill(make_req):
    s = _sched()
    s.on_new_request(make_req())
    assert len(s._heartbeat_by_engine) == 0


# ===================================================================
# Scheduler: _stop_heartbeat
# ===================================================================


def test_stop_heartbeat_partial_and_full():
    """Stop one of two reqs on same engine, then stop the other."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    s._stop_heartbeat("id-1")
    assert s._heartbeat_by_engine[_ENGINE_A].req_ids == {"prefill-2"}
    assert "id-1" not in s._heartbeat_req_engine

    s._stop_heartbeat("id-2")
    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Scheduler: build_connector_meta throttling
# ===================================================================


def test_build_connector_meta_heartbeat_throttling():
    # kv_lease_duration=30 => _heartbeat_interval = 30 // 6 = 5
    s = _sched(kv_lease_duration=30)
    s.on_new_request(_req(1))

    # Ensure the first call triggers by placing last_heartbeat far in the past.
    s._last_heartbeat_time = time.perf_counter() - 10
    meta1 = s.build_connector_meta(MagicMock())
    assert _ENGINE_A in meta1.heartbeat_by_engine

    # Immediate second call is throttled (< 5s since last).
    meta2 = s.build_connector_meta(MagicMock())
    assert len(meta2.heartbeat_by_engine) == 0


# ===================================================================
# Scheduler: cleanup paths (update_connector_output / request_finished)
# ===================================================================


def test_update_connector_output_stops_heartbeat():
    s = _sched()
    s.on_new_request(_req(1))

    s.update_connector_output(
        KVConnectorOutput(
            finished_sending=None,
            finished_recving={"id-1"},
            invalid_block_ids=set(),
        )
    )

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


def test_request_finished_stops_heartbeat():
    s = _sched()
    r = _req(1)
    s.on_new_request(r)

    # Simulate update_state_after_alloc having consumed do_remote_prefill.
    r.kv_transfer_params["do_remote_prefill"] = False
    s.request_finished(r, block_ids=())

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Worker: _handle_heartbeat
# ===================================================================


def test_handle_heartbeat():
    w = _worker_stub()
    far_future = time.perf_counter() + 99999
    w._reqs_to_send = {"req-a": 100.0, "req-b": far_future}

    before = time.perf_counter()
    w._handle_heartbeat("req-a,req-b,req-unknown")

    # req-a: pushed forward to ~now+20.
    assert w._reqs_to_send["req-a"] >= before + 20
    # req-b: already far out, max() keeps it.
    assert w._reqs_to_send["req-b"] >= far_future
    # req-unknown: not added.
    assert "req-unknown" not in w._reqs_to_send


def test_send_heartbeat_preserves_all_parallel_sizes():
    w = _worker_stub()
    w._hb_handshake_notif_only = True
    w._ensure_handshake = MagicMock(return_value=MagicMock())
    metadata = NixlConnectorMetadata()
    metadata.heartbeat_by_engine[_ENGINE_A] = HeartbeatInfo(
        req_ids={"prefill-1"},
        host="my-host",
        port=1234,
        tp_size=4,
        dcp_size=2,
        pcp_size=2,
        pp_size=3,
    )

    w._send_heartbeats(metadata)

    w._ensure_handshake.assert_called_once_with(
        _ENGINE_A,
        "my-host",
        1234,
        4,
        2,
        2,
        3,
        True,
    )
