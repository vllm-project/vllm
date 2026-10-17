# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for mori-io write_done routing (#51681)."""

import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOMode,
    RemoteAllocInfo,
    ReqMeta,
    WriteTask,
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)

from .utils import make_moriio_writer

pytestmark = pytest.mark.cpu_test


def _fake_worker(
    alloc: dict[str, Any],
    *,
    tp_rank: int = 0,
    set_remote_dp_size_local: int | None = None,
) -> Any:
    wrapper = SimpleNamespace(
        done_remote_allocate_req_dict=alloc,
        lock=threading.Lock(),
        done_req_ids=[],
        sent_notifies=[],
    )
    wrapper.send_notify = lambda transfer_id, remote_ip, remote_port, message_type: (
        wrapper.sent_notifies.append(
            (transfer_id, remote_ip, remote_port, message_type)
        )
    )
    wrapper.waiting_for_transfer_complete = lambda statuses: None
    wrapper._mark_transfer_terminal_locked = lambda transfer_id: None
    wrapper._is_transfer_terminal_locked = lambda transfer_id: False
    worker = SimpleNamespace(moriio_wrapper=wrapper, tp_rank=tp_rank)
    if set_remote_dp_size_local is not None:
        worker.remote_dp_size_local = set_remote_dp_size_local
    return worker


def _req_meta(
    *,
    transfer_id: str = "tA",
    remote_host: str = "10.0.0.1",
    multi_pod_hosts: list[str] | None = None,
    remote_dp_size: int = 16,
    remote_dp_size_local: int = 8,
) -> ReqMeta:
    return ReqMeta(
        transfer_id=transfer_id,
        local_block_ids=[[1]],
        remote_block_ids=[[2]],
        remote_host=remote_host,
        remote_port=1,
        remote_handshake_port=2,
        remote_notify_port=20000,
        remote_engine_id="remote-engine",
        tp_size=1,
        remote_dp_size=remote_dp_size,
        multi_pod_hosts=list(multi_pod_hosts or []),
        remote_dp_size_local=remote_dp_size_local,
    )


def _write_task(
    transfer_id: str,
    remote_ip: str,
    *,
    multi_pod_hosts: list[str] | None = None,
    remote_dp_size_local: int = 0,
    remote_dp_size: int = 1,
    request_id: str = "req",
) -> WriteTask:
    return WriteTask(
        request_id=request_id,
        transfer_id=transfer_id,
        dst_engine_id="remote-engine",
        local_block_ids=[1, 3],
        remote_block_ids_hint=None,
        layer_name="layer0",
        event=None,
        remote_notify_port=20000,
        remote_ip=remote_ip,
        multi_pod_hosts=multi_pod_hosts or [],
        remote_dp_size_local=remote_dp_size_local,
        remote_dp_size=remote_dp_size,
    )


def _real_wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_req_ids = []
    wrapper.done_remote_allocate_req_dict = {}
    wrapper.done_write_cache_req_ids = []
    wrapper.failed_write_cache_req_ids = []
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


def test_execute_write_task_ip_is_per_task_under_overwrite():
    alloc = {"tA": RemoteAllocInfo(block_ids=None, decode_dp_rank=0)}
    worker = _fake_worker(alloc)
    worker.multi_pod_hosts = ["10.0.0.2"]
    worker.remote_dp_size_local = 1

    writer = make_moriio_writer(worker)
    task_a = _write_task(
        "tA", "10.0.0.1", multi_pod_hosts=["10.0.0.1"], remote_dp_size_local=1
    )
    writer._execute_write_task(task_a)

    assert alloc["tA"].completion_remote_ip == "10.0.0.1"
    assert alloc["tA"].completion_remote_ip != "10.0.0.2"


def test_deferred_task_ip_is_per_task_under_overwrite():
    alloc = {"tA": RemoteAllocInfo(block_ids=None, decode_dp_rank=0)}
    worker = _fake_worker(alloc)
    worker.multi_pod_hosts = ["10.0.0.2"]
    worker.remote_dp_size_local = 1

    writer = make_moriio_writer(worker)
    task_a = _write_task(
        "tA", "10.0.0.1", multi_pod_hosts=["10.0.0.1"], remote_dp_size_local=1
    )
    writer._deferred_tasks.append(task_a)
    writer._process_deferred_tasks()

    assert alloc["tA"].completion_remote_ip == "10.0.0.1"
    assert writer._deferred_tasks == []


def test_two_requests_resolve_distinct_hosts():
    alloc = {
        "tA": RemoteAllocInfo(block_ids=None, decode_dp_rank=0),
        "tB": RemoteAllocInfo(block_ids=None, decode_dp_rank=3),
    }
    worker = _fake_worker(alloc)
    writer = make_moriio_writer(worker)

    task_a = _write_task(
        "tA", "10.0.0.1", multi_pod_hosts=["10.0.0.1"], remote_dp_size_local=1
    )
    task_b = _write_task(
        "tB", "10.0.0.2", multi_pod_hosts=["10.0.0.2"], remote_dp_size_local=1
    )
    worker.multi_pod_hosts = ["10.0.0.2"]
    writer._execute_write_task(task_a)
    writer._execute_write_task(task_b)

    assert alloc["tA"].completion_remote_ip == "10.0.0.1"
    assert alloc["tB"].completion_remote_ip == "10.0.0.2"


def test_finalize_notify_port_uses_pod_local_rank():
    base = 20000
    info = RemoteAllocInfo(block_ids=None, decode_dp_rank=9)
    alloc = {"tA": info}
    worker = _fake_worker(alloc, tp_rank=0)
    writer = make_moriio_writer(worker)

    task = _write_task(
        "tA",
        "10.0.0.1",
        multi_pod_hosts=["10.0.0.1", "10.0.0.2"],
        remote_dp_size_local=8,
    )
    writer._execute_write_task(task)
    assert info.completion_remote_ip == "10.0.0.2"

    info.writes_expected = 1
    info.writes_done = 1
    writer._finalize_if_complete("tA", info)

    assert worker.moriio_wrapper.sent_notifies
    _tid, ip, port, mtype = worker.moriio_wrapper.sent_notifies[-1]
    assert mtype == "write_done"
    assert ip == "10.0.0.2"
    assert port == base + get_port_offset(9 % 8, 0)


def test_finalize_port_ignores_stale_worker_dp_local():
    base = 20000
    info = RemoteAllocInfo(block_ids=None, decode_dp_rank=9)
    alloc = {"tA": info}
    worker = _fake_worker(alloc, tp_rank=0, set_remote_dp_size_local=1)
    writer = make_moriio_writer(worker)

    task = _write_task(
        "tA",
        "10.0.0.1",
        multi_pod_hosts=["10.0.0.1", "10.0.0.2"],
        remote_dp_size_local=8,
    )
    writer._execute_write_task(task)
    assert info.completion_notify_port == base + get_port_offset(9 % 8, 0)

    info.writes_expected = 1
    info.writes_done = 1
    writer._finalize_if_complete("tA", info)

    _tid, _ip, port, mtype = worker.moriio_wrapper.sent_notifies[-1]
    assert mtype == "write_done"
    assert port == base + get_port_offset(9 % 8, 0)
    assert port != base + get_port_offset(9 % 1, 0)


def test_finalize_single_pod_port_uses_global_rank():
    base = 20000
    info = RemoteAllocInfo(block_ids=None, decode_dp_rank=3)
    alloc = {"tA": info}
    worker = _fake_worker(alloc, tp_rank=2)
    writer = make_moriio_writer(worker)

    task = _write_task("tA", "10.0.0.1", multi_pod_hosts=[], remote_dp_size_local=0)
    writer._execute_write_task(task)
    assert info.completion_remote_ip == "10.0.0.1"
    assert info.completion_notify_port == base + get_port_offset(3, 2)

    info.writes_expected = 1
    info.writes_done = 1
    writer._finalize_if_complete("tA", info)

    _tid, _ip, port, _mtype = worker.moriio_wrapper.sent_notifies[-1]
    assert port == base + get_port_offset(3, 2)


def test_write_blocks_for_req_forwards_routing_to_schedule_write():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    captured: dict[str, Any] = {}

    def _capture_schedule_write_blocks(**kwargs: Any) -> None:
        captured.update(kwargs)

    worker.schedule_write_blocks = _capture_schedule_write_blocks  # type: ignore[method-assign]

    meta = _req_meta(
        remote_host="10.0.0.1",
        multi_pod_hosts=["10.0.0.1", "10.0.0.2"],
        remote_dp_size_local=8,
    )
    worker._write_blocks_for_req("req-a", meta, "layer0", kv_layer=None)

    assert captured["multi_pod_hosts"] == ["10.0.0.1", "10.0.0.2"]
    assert captured["remote_dp_size_local"] == 8
    assert captured["remote_ip"] == "10.0.0.1"
    assert captured["local_block_ids"] == [1]
    assert captured["remote_block_ids"] == [2]
    assert not hasattr(worker, "multi_pod_hosts")


def test_write_blocks_for_req_falls_back_when_multi_pod_hosts_empty():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    captured: dict[str, Any] = {}

    def _capture_schedule_write_blocks(**kwargs: Any) -> None:
        captured.update(kwargs)

    worker.schedule_write_blocks = _capture_schedule_write_blocks  # type: ignore[method-assign]

    meta = _req_meta(
        remote_host="10.0.0.9",
        multi_pod_hosts=[],
        remote_dp_size=4,
        remote_dp_size_local=0,
    )
    worker._write_blocks_for_req("req-b", meta, "layer0", kv_layer=None)

    assert captured["multi_pod_hosts"] == ["10.0.0.9"]
    assert captured["remote_dp_size_local"] == 4
    assert captured["local_block_ids"] == [1]
    assert captured["remote_block_ids"] == [2]


def test_deferred_write_timeout_fails_transfer_on_every_decode_rank():
    wrapper = _real_wrapper()
    sent: list[tuple[str, int, str]] = []
    wrapper.send_notify = lambda tid, ip, port, message_type: sent.append(
        (ip, port, message_type)
    )
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper, tp_rank=0))
    task = _write_task(
        "tA",
        "10.0.0.1",
        multi_pod_hosts=["10.0.0.1", "10.0.0.2"],
        remote_dp_size_local=2,
        remote_dp_size=4,
    )
    task.enqueue_time = time.perf_counter() - 2 * writer._defer_timeout
    writer._deferred_tasks.append(task)

    writer._process_deferred_tasks()

    assert writer._deferred_tasks == []
    assert [ack.transfer_id for ack in wrapper.done_req_ids] == ["tA"]
    # Global DP ranks 0-3, two per pod: the owning rank is unknown.
    assert sent == [
        (host, 20000 + get_port_offset(local_rank, 0), "write_failed")
        for host in ("10.0.0.1", "10.0.0.2")
        for local_rank in (0, 1)
    ]

    # A late decode allocation must not resurrect a write into freed blocks.
    with patch.object(moriio_engine, "get_role", return_value=ROLE.PRODUCER):
        wrapper._handle_remote_blocks_message(
            {"transfer_id": "tA", "block_notify_list": [7], "decode_rank": 1}
        )
    assert wrapper.done_remote_allocate_req_dict == {}


def test_write_scheduler_waits_for_worker_ack_past_defer_deadline():
    scheduler = SimpleNamespace(
        is_producer=True,
        mode=MoRIIOMode.WRITE,
        _defer_timeout=60.0,
        _deferred_send_deadlines={"req-1": (time.monotonic() - 1.0, "tA")},
        _pending_sent_acks={},
        unmap_request_id=lambda *args, **kwargs: None,
    )
    output = SimpleNamespace(finished_sending=set())

    MoRIIOConnectorScheduler.update_connector_output(scheduler, output)
    assert output.finished_sending is None
    assert "req-1" in scheduler._deferred_send_deadlines

    output.finished_sending = {"req-1"}
    MoRIIOConnectorScheduler.update_connector_output(scheduler, output)
    assert output.finished_sending == {"req-1"}
    assert scheduler._deferred_send_deadlines == {}


def test_write_failed_is_reported_as_failed_recving_on_owning_rank():
    wrapper = _real_wrapper()
    with patch.object(moriio_engine, "get_role", return_value=ROLE.CONSUMER):
        for tid in ("tA", "tOtherRank"):
            wrapper._handle_structured_message(
                {"type": "write_failed", "transfer_id": tid}
            )
    worker = SimpleNamespace(
        mode=MoRIIOMode.WRITE,
        is_producer=False,
        moriio_wrapper=wrapper,
        transfer_id_to_request_id={"tA": "req-1"},
        _unmatched_write_failures=OrderedDict(),
        get_finished=lambda: (set(), set()),
    )

    results = MoRIIOConnectorWorker.get_transfer_results(worker)

    assert results.finished_recving == {"req-1"}
    assert results.failed_recving == {"req-1"}
    assert wrapper.pop_finished_write_req_ids() == set()
    assert list(worker._unmatched_write_failures) == ["tOtherRank"]
