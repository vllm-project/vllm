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
    get_moriio_write_target_tp_ranks,
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
        waited_statuses=[],
    )
    wrapper.send_notify = lambda transfer_id, remote_ip, remote_port, message_type: (
        wrapper.sent_notifies.append(
            (transfer_id, remote_ip, remote_port, message_type)
        )
    )
    wrapper.waiting_for_transfer_complete = lambda statuses: (
        wrapper.waited_statuses.append(list(statuses))
    )
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
    tp_size: int = 1,
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
        tp_size=tp_size,
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
    target_tp_ranks: tuple[int, ...] = (),
    remote_tp_size: int = 0,
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
        target_tp_ranks=target_tp_ranks,
        remote_tp_size=remote_tp_size,
    )


@pytest.mark.parametrize(
    (
        "producer_tp_rank",
        "producer_tp_size",
        "consumer_tp_size",
        "is_mla",
        "expected",
    ),
    [
        (0, 2, 4, True, (0, 1)),
        (1, 2, 4, True, (2, 3)),
        (0, 4, 8, True, (0, 1)),
        (1, 4, 8, True, (2, 3)),
        (2, 4, 8, True, (4, 5)),
        (3, 4, 8, True, (6, 7)),
        (2, 4, 4, True, (2,)),
        (0, 4, 2, True, (0,)),
        (3, 4, 2, True, (1,)),
        (0, 2, 4, False, (0,)),
        (1, 2, 4, False, (2,)),
    ],
)
def test_get_moriio_write_target_tp_ranks(
    producer_tp_rank: int,
    producer_tp_size: int,
    consumer_tp_size: int,
    is_mla: bool,
    expected: tuple[int, ...],
):
    assert (
        get_moriio_write_target_tp_ranks(
            producer_tp_rank,
            producer_tp_size,
            consumer_tp_size,
            is_mla,
        )
        == expected
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


def test_execute_write_task_writes_every_fanout_target():
    info = RemoteAllocInfo(block_ids=[2], decode_dp_rank=0)
    alloc = {"tA": info}
    worker = _fake_worker(alloc)
    session_engine_ids: list[str] = []

    worker.get_engine_name_with_dp_tp = (
        lambda engine_id, dp_rank, tp_rank: f"{engine_id}_dp{dp_rank}_tp{tp_rank}"
    )

    def _get_built_session(engine_id: str):
        session_engine_ids.append(engine_id)
        return [engine_id], f"metadata-{engine_id}"

    worker._get_built_session = _get_built_session
    writer = make_moriio_writer(worker)
    write_sessions: list[list[str]] = []

    def _prepare_transfer_plan(
        task, request_info, remote_moriio_meta, remote_engine_id=None
    ):
        return "plan"

    def _do_layer_write(plan, sessions):
        write_sessions.append(sessions)
        return [f"status-{sessions[0]}"]

    writer._prepare_transfer_plan = _prepare_transfer_plan
    writer._do_layer_write = _do_layer_write

    task = _write_task(
        "tA",
        "10.0.0.1",
        target_tp_ranks=(2, 3),
        remote_tp_size=4,
    )
    task.event = SimpleNamespace(synchronize=lambda: None)
    writer._execute_write_task(task)

    assert session_engine_ids == [
        "remote-engine_dp0_tp2",
        "remote-engine_dp0_tp3",
    ]
    assert write_sessions == [
        ["remote-engine_dp0_tp2"],
        ["remote-engine_dp0_tp3"],
    ]
    assert info.transfer_statuses == [
        "status-remote-engine_dp0_tp2",
        "status-remote-engine_dp0_tp3",
    ]


def test_fanout_write_done_notifies_every_target_after_completion():
    base = 20000
    info = RemoteAllocInfo(block_ids=None, decode_dp_rank=1)
    alloc = {"tA": info}
    worker = _fake_worker(alloc, tp_rank=0)
    writer = make_moriio_writer(worker)

    task = _write_task(
        "tA",
        "10.0.0.1",
        target_tp_ranks=(2, 3),
        remote_tp_size=4,
    )
    writer._execute_write_task(task)

    assert info.completion_notify_targets == {
        ("10.0.0.1", base + get_port_offset(1, 2, 4)),
        ("10.0.0.1", base + get_port_offset(1, 3, 4)),
    }

    info.writes_expected = 1
    info.transfer_statuses.extend(["write-to-tp2", "write-to-tp3"])
    writer._finalize_if_complete("tA", info)
    assert worker.moriio_wrapper.waited_statuses == []
    assert worker.moriio_wrapper.sent_notifies == []

    info.writes_done = 1
    writer._finalize_if_complete("tA", info)

    assert worker.moriio_wrapper.waited_statuses == [
        ["write-to-tp2", "write-to-tp3"]
    ]
    assert worker.moriio_wrapper.sent_notifies == [
        (
            "tA",
            "10.0.0.1",
            base + get_port_offset(1, 2, 4),
            "write_done",
        ),
        (
            "tA",
            "10.0.0.1",
            base + get_port_offset(1, 3, 4),
            "write_done",
        ),
    ]


def test_write_blocks_for_req_forwards_routing_to_schedule_write():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 0
    worker.world_size = 1
    worker.use_mla = False
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
    worker.tp_rank = 0
    worker.world_size = 1
    worker.use_mla = False
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


def test_write_blocks_for_req_schedules_mla_write_fanout():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 1
    worker.world_size = 2
    worker.use_mla = True
    captured: dict[str, Any] = {}

    def _capture_schedule_write_blocks(**kwargs: Any) -> None:
        captured.update(kwargs)

    worker.schedule_write_blocks = _capture_schedule_write_blocks  # type: ignore[method-assign]

    meta = _req_meta(tp_size=4)
    worker._write_blocks_for_req("req-c", meta, "layer0", kv_layer=None)

    assert captured["target_tp_ranks"] == (2, 3)
    assert captured["remote_tp_size"] == 4


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


def test_deferred_write_timeout_notifies_every_fanout_target():
    wrapper = _real_wrapper()
    sent: list[tuple[str, int, str]] = []
    wrapper.send_notify = lambda tid, ip, port, message_type: sent.append(
        (ip, port, message_type)
    )
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper, tp_rank=0))
    task = _write_task(
        "tA",
        "10.0.0.1",
        remote_dp_size_local=2,
        remote_dp_size=2,
        target_tp_ranks=(2, 3),
        remote_tp_size=4,
    )
    task.enqueue_time = time.perf_counter() - 2 * writer._defer_timeout
    writer._deferred_tasks.append(task)

    writer._process_deferred_tasks()

    assert sent == [
        (
            "10.0.0.1",
            20000 + get_port_offset(dp_rank, tp_rank, 4),
            "write_failed",
        )
        for dp_rank in (0, 1)
        for tp_rank in (2, 3)
    ]


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
