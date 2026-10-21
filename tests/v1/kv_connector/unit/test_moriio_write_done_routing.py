# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for mori-io write_done routing (#51681)."""

import queue
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
    LayerTransferPlan,
    MoRIIOConnectorMetadata,
    MoRIIOError,
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
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_tp_mapping import (
    MoRIIOWriteTPMapping,
    compute_write_tp_mapping,
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
        sent_fields=[],
        waited_statuses=[],
    )

    def send_notify(
        transfer_id, remote_ip, remote_port, message_type, message_fields=None
    ):
        wrapper.sent_notifies.append(
            (transfer_id, remote_ip, remote_port, message_type)
        )
        wrapper.sent_fields.append(message_fields)

    wrapper.send_notify = send_notify
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
    tp_mapping: MoRIIOWriteTPMapping | None = None,
) -> WriteTask:
    return WriteTask(
        request_id=request_id,
        transfer_id=transfer_id,
        dst_engine_id="remote-engine",
        local_block_ids=[[1, 3]],
        remote_block_ids_hint=None,
        layer_name="layer0",
        event=None,
        remote_notify_port=20000,
        remote_ip=remote_ip,
        multi_pod_hosts=multi_pod_hosts or [],
        remote_dp_size_local=remote_dp_size_local,
        remote_dp_size=remote_dp_size,
        remote_tp_size=None if tp_mapping is None else tp_mapping.consumer_tp_size,
        tp_mapping=tp_mapping,
    )


# Prefill rank 1 of TP2 writing an MLA latent to decode TP4: ranks 2 and 3.
MLA_FANOUT = compute_write_tp_mapping(1, 2, 4, 1, True, False)


def _real_wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_req_ids = []
    wrapper.done_remote_allocate_req_dict = {}
    wrapper.done_write_cache_req_ids = []
    wrapper._write_done_counts = {}
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
    info = RemoteAllocInfo(block_ids=[[2]], decode_dp_rank=0)
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
    worker._is_mamba_layer = lambda layer_name: False
    writer = make_moriio_writer(worker)
    write_sessions: list[list[str]] = []

    def _prepare_transfer_plan(
        task, request_info, remote_moriio_meta, remote_engine_id=None, target=None
    ):
        return "plan"

    def _do_layer_write(plan, sessions):
        write_sessions.append(sessions)
        return [f"status-{sessions[0]}"]

    writer._prepare_transfer_plan = _prepare_transfer_plan
    writer._do_layer_write = _do_layer_write

    task = _write_task("tA", "10.0.0.1", tp_mapping=MLA_FANOUT)
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

    task = _write_task("tA", "10.0.0.1", tp_mapping=MLA_FANOUT)
    writer._execute_write_task(task)

    assert info.completion_notify_targets == {
        ("10.0.0.1", base + get_port_offset(1, 2)): 1,
        ("10.0.0.1", base + get_port_offset(1, 3)): 1,
    }

    info.writes_expected = 1
    info.transfer_statuses.extend(["write-to-tp2", "write-to-tp3"])
    writer._finalize_if_complete("tA", info)
    assert worker.moriio_wrapper.waited_statuses == []
    assert worker.moriio_wrapper.sent_notifies == []

    info.writes_done = 1
    writer._finalize_if_complete("tA", info)

    assert worker.moriio_wrapper.waited_statuses == [["write-to-tp2", "write-to-tp3"]]
    assert worker.moriio_wrapper.sent_notifies == [
        (
            "tA",
            "10.0.0.1",
            base + get_port_offset(1, 2),
            "write_done",
        ),
        (
            "tA",
            "10.0.0.1",
            base + get_port_offset(1, 3),
            "write_done",
        ),
    ]


def test_write_blocks_for_req_forwards_routing_to_schedule_write():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 0
    worker.world_size = 1
    worker.use_mla = False
    worker._has_mamba = False
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: 8)
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
    assert captured["local_block_ids"] == [[1]]
    assert captured["remote_block_ids"] == [[2]]
    assert not hasattr(worker, "multi_pod_hosts")


def test_write_blocks_for_req_falls_back_when_multi_pod_hosts_empty():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 0
    worker.world_size = 1
    worker.use_mla = False
    worker._has_mamba = False
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: 8)
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
    assert captured["local_block_ids"] == [[1]]
    assert captured["remote_block_ids"] == [[2]]


def test_write_blocks_for_req_schedules_mla_write_fanout():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 1
    worker.world_size = 2
    worker.use_mla = True
    worker._has_mamba = False
    captured: dict[str, Any] = {}

    def _capture_schedule_write_blocks(**kwargs: Any) -> None:
        captured.update(kwargs)

    worker.schedule_write_blocks = _capture_schedule_write_blocks  # type: ignore[method-assign]

    meta = _req_meta(tp_size=4)
    worker._write_blocks_for_req("req-c", meta, "layer0", kv_layer=None)

    assert captured["tp_mapping"].target_tp_ranks == (2, 3)
    assert captured["remote_tp_size"] == 4


class _FanoutSaveWorker(MoRIIOConnectorWorker):
    handshakes: list[str]
    writes: list[str]

    def _background_moriio_handshake(
        self, req_id: str, remote_engine_id: str, meta: ReqMeta
    ) -> None:
        self.handshakes.append(req_id)

    def _write_blocks_for_req(self, req_id: str, meta: ReqMeta, layer_name, kv_layer):
        self.writes.append(req_id)


def _fanout_save_worker(remote_agents: set[str]) -> _FanoutSaveWorker:
    worker = _FanoutSaveWorker.__new__(_FanoutSaveWorker)
    worker.is_producer = True
    worker.mode = MoRIIOMode.WRITE
    worker.tp_rank = 0
    worker.world_size = 1
    worker.use_mla = True
    worker._has_mamba = False
    worker._remote_agents = {engine_id: {"agent"} for engine_id in remote_agents}
    # Set by the handshake callback even when one of the targets failed.
    worker.write_ready_flags = {"10.0.0.1:2": True}
    worker._handshake_lock = threading.Lock()
    worker._ready_requests = queue.Queue()
    worker.moriio_config = SimpleNamespace(transfer_timeout=0.01)
    worker.handshakes = []
    worker.writes = []
    return worker


def test_fanout_write_retries_handshake_after_one_target_failed():
    worker = _fanout_save_worker({"10.0.0.1:2_dp0_tp0"})
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_save["req"] = _req_meta(tp_size=2)

    worker.save_kv_layer(metadata, "layer0", kv_layer=None, attn_metadata=None)

    assert worker.handshakes == ["req"]
    assert worker.writes == []


def test_fanout_write_proceeds_once_every_target_session_exists():
    worker = _fanout_save_worker({"10.0.0.1:2_dp0_tp0", "10.0.0.1:2_dp0_tp1"})
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_save["req"] = _req_meta(tp_size=2)

    worker.save_kv_layer(metadata, "layer0", kv_layer=None, attn_metadata=None)

    assert worker.handshakes == []
    assert worker.writes == ["req"]


def test_rank_with_nothing_to_write_does_not_wait_for_a_handshake():
    # MLA fan-in: prefill rank 1 of TP2 holds the same latent as rank 0, which
    # writes it to the single decode rank.
    worker = _fanout_save_worker(set())
    worker.tp_rank = 1
    worker.world_size = 2
    worker.write_ready_flags = {}
    worker.moriio_config = SimpleNamespace(transfer_timeout=30.0)
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_save["req"] = _req_meta(tp_size=1)

    start = time.monotonic()
    worker.save_kv_layer(metadata, "layer0", kv_layer=None, attn_metadata=None)

    assert time.monotonic() - start < 5
    assert worker.handshakes == []
    assert worker.writes == ["req"]


@pytest.mark.parametrize(
    ("params", "expected_tp_ranks"),
    [
        ({"remote_tp_size": 2}, [0, 1]),
        ({"tp_size": 8}, list(range(8))),
        ({}, list(range(4))),
    ],
)
def test_release_reaches_every_prefill_rank_and_no_other(params, expected_tp_ranks):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.tp_size = 4
    sent: list[tuple[str, str, int]] = []
    scheduler._send_transfer_release = (  # type: ignore[method-assign]
        lambda transfer_id, host, port: sent.append((transfer_id, host, port))
    )

    scheduler._release_write_prefill_blocks(
        "req",
        {
            "transfer_id": "tA",
            "remote_dp_rank": 0,
            "remote_host": "10.0.0.1",
            "remote_notify_port": 20000,
            **params,
        },
    )

    assert sent == [
        ("tA", "10.0.0.1", 20000 + get_port_offset(0, tp_rank))
        for tp_rank in expected_tp_ranks
    ]


def test_deferred_write_timeout_fails_transfer_on_every_decode_rank(
    *, monkeypatch: pytest.MonkeyPatch
):
    wrapper = _real_wrapper()
    sent: list[tuple[str, int, str]] = []
    monkeypatch.setattr(
        wrapper,
        "send_notify",
        lambda tid, ip, port, message_type: sent.append((ip, port, message_type)),
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


def test_deferred_write_timeout_notifies_every_fanout_target(
    *, monkeypatch: pytest.MonkeyPatch
):
    wrapper = _real_wrapper()
    sent: list[tuple[str, int, str]] = []
    monkeypatch.setattr(
        wrapper,
        "send_notify",
        lambda tid, ip, port, message_type: sent.append((ip, port, message_type)),
    )
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper, tp_rank=0))
    task = _write_task(
        "tA",
        "10.0.0.1",
        remote_dp_size_local=2,
        remote_dp_size=2,
        tp_mapping=MLA_FANOUT,
    )
    task.enqueue_time = time.perf_counter() - 2 * writer._defer_timeout
    writer._deferred_tasks.append(task)

    writer._process_deferred_tasks()

    assert sent == [
        (
            "10.0.0.1",
            20000 + get_port_offset(dp_rank, tp_rank),
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


def test_write_rejects_unmappable_tp_before_scheduling_any_layer(
    *, monkeypatch: pytest.MonkeyPatch
):
    from unittest.mock import Mock

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = True
    worker.use_mla = False
    worker.tp_rank = 0
    worker.world_size = 2
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: 8)
    monkeypatch.setattr(worker, "schedule_write_blocks", Mock())

    with pytest.raises(ValueError, match="multiple of the other"):
        worker._write_blocks_for_req(
            "req", _req_meta(tp_size=3), "attention", kv_layer=None
        )

    assert isinstance(worker.schedule_write_blocks, Mock)
    worker.schedule_write_blocks.assert_not_called()


@pytest.mark.parametrize(
    "notify,expected",
    [
        # Plain WRITE peers still send one flat attention list.
        ([7, 8], [[7, 8]]),
        # Hybrid WRITE sends one list per transfer group, kept as-is even
        # when the first group is empty (full attention hit, state only).
        ([[7], [9]], [[7], [9]]),
        ([[], [9]], [[], [9]]),
    ],
)
def test_remote_blocks_message_normalizes_to_grouped_block_ids(notify, expected):
    wrapper = _real_wrapper()
    with patch.object(moriio_engine, "get_role", return_value=ROLE.PRODUCER):
        wrapper._handle_remote_blocks_message(
            {"transfer_id": "tA", "block_notify_list": notify, "decode_rank": 2}
        )
    info = wrapper.done_remote_allocate_req_dict["tA"]
    assert info.block_ids == expected
    assert info.decode_dp_rank == 2


def test_empty_layer_plan_posts_no_rdma_write():
    from unittest.mock import Mock

    wrapper = SimpleNamespace(write_remote_data=Mock(), write_remote_data_single=Mock())
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper))
    plan = LayerTransferPlan(
        request_id="req",
        transfer_id="tA",
        layer_name="layer0",
        sess_idx=0,
        transfer_local_offsets=[],
        transfer_remote_offsets=[],
        transfer_sizes=[],
    )

    assert writer._do_layer_write(plan, ["session"]) == []
    wrapper.write_remote_data.assert_not_called()
    wrapper.write_remote_data_single.assert_not_called()


@pytest.mark.parametrize("local_state", [[], [1, 2]])
def test_hybrid_write_requires_exactly_one_running_state_slot(local_state):
    worker = SimpleNamespace(
        _mamba_blocks_for_layer=lambda layer_name, block_ids: block_ids[1],
    )
    writer = make_moriio_writer(worker)
    task = _write_task("tA", "10.0.0.1")
    task.layer_name = "kda"
    task.local_block_ids = [[], local_state]

    with pytest.raises(MoRIIOError, match="one running-state slot"):
        writer._prepare_mamba_transfer_plans(task, RemoteAllocInfo(block_ids=[[], [9]]))
