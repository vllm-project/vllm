# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRI-IO WRITE producer and a consumer that restarts.

The producer keys a consumer by host:handshake port, which a restarted consumer
keeps, so without an incarnation check it would keep writing over sessions to
memory and QPs that no longer exist."""

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
    MoRIIOAgentMetadata,
    RemoteAllocInfo,
    StalePeerError,
    TransferError,
    WriteTask,
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)

from ..utils import make_moriio_writer

pytestmark = pytest.mark.cpu_test

PEER = "10.0.0.1:2"  # host:handshake port, what the producer keys a consumer by
NOTIFY_PORT = 20000


def _wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_req_ids = []
    wrapper.done_remote_allocate_req_dict = {}
    wrapper.done_write_cache_req_ids = []
    wrapper._write_done_senders = {}
    wrapper.failed_write_cache_req_ids = []
    wrapper._terminal_transfer_ids = OrderedDict()
    wrapper.sent = []
    wrapper.send_notify = lambda tid, ip, port, message_type, **kw: wrapper.sent.append(
        (tid, ip, port, message_type)
    )
    return wrapper


def _producer(peer_incarnations: dict[str, str]) -> Any:
    """A producer worker whose sessions were built for the given incarnations
    (session key -> incarnation); records what it is asked to drop."""
    worker = SimpleNamespace(
        moriio_wrapper=_wrapper(),
        tp_rank=0,
        world_size=1,
        pp_rank=0,
        pp_size=1,
        dropped=[],
        sessions_used=[],
    )
    worker.peer_incarnation = peer_incarnations.get
    worker.get_engine_name_with_dp = lambda engine, dp: f"{engine}_dp{dp}"

    def invalidate_remote_peer(remote_engine_id, reason, current_incarnation=""):
        worker.dropped.append((remote_engine_id, current_incarnation))
        for eid in list(peer_incarnations):
            if peer_incarnations[eid] != current_incarnation:
                del peer_incarnations[eid]

    worker.invalidate_remote_peer = invalidate_remote_peer

    def _get_built_session(engine_id):
        worker.sessions_used.append(engine_id)
        return [engine_id], "remote-meta"

    worker._get_built_session = _get_built_session
    worker._is_mamba_layer = lambda layer_name: False
    return worker


def _task(transfer_id: str = "tA", age: float = 0.0) -> WriteTask:
    task = WriteTask(
        request_id="req",
        transfer_id=transfer_id,
        dst_engine_id=PEER,
        local_block_ids=[[1]],
        remote_block_ids_hint=None,
        layer_name="layer0",
        event=SimpleNamespace(synchronize=lambda: None),
        remote_notify_port=NOTIFY_PORT,
        remote_ip="10.0.0.1",
        multi_pod_hosts=[],
        remote_dp_size_local=2,
        remote_dp_size=2,
        remote_tp_size=None,
        tp_mapping=None,
    )
    task.enqueue_time = time.perf_counter() - age
    return task


def _writer(worker: Any, writes: list) -> Any:
    writer = make_moriio_writer(worker)
    writer._prepare_transfer_plan = lambda *a, **kw: "plan"

    def _do_layer_write(plan, sessions):
        writes.append(sessions[0])
        return []

    writer._do_layer_write = _do_layer_write
    writer._mark_write_done = lambda transfer_id, info: None
    return writer


def test_remote_blocks_message_keeps_the_consumer_incarnation():
    wrapper = _wrapper()
    with patch.object(moriio_engine, "get_role", return_value=ROLE.PRODUCER):
        wrapper._handle_remote_blocks_message(
            {
                "transfer_id": "tA",
                "block_notify_list": [7],
                "decode_rank": 1,
                "incarnation": "new",
            }
        )
        wrapper._handle_remote_blocks_message(
            {"transfer_id": "tB", "block_notify_list": [8], "decode_rank": 0}
        )
    assert wrapper.done_remote_allocate_req_dict["tA"].incarnation == "new"
    # A consumer that does not send one is not checked.
    assert wrapper.done_remote_allocate_req_dict["tB"].incarnation == ""


def test_write_for_the_same_incarnation_proceeds():
    worker = _producer({f"{PEER}_dp1": "inc-1"})
    worker.moriio_wrapper.done_remote_allocate_req_dict["tA"] = RemoteAllocInfo(
        block_ids=[[2]], decode_dp_rank=1, incarnation="inc-1"
    )
    writes: list = []
    writer = _writer(worker, writes)
    writer._deferred_tasks.append(_task())

    writer._process_deferred_tasks()

    assert writes == [f"{PEER}_dp1"]
    assert worker.dropped == []
    assert worker.moriio_wrapper.sent == []


def test_write_to_a_restarted_consumer_fails_fast_and_drops_its_sessions():
    # Sessions to DP rank 1 were built for "old"; the blocks come from "new".
    worker = _producer({f"{PEER}_dp1": "old"})
    worker.moriio_wrapper.done_remote_allocate_req_dict["tA"] = RemoteAllocInfo(
        block_ids=[[2]], decode_dp_rank=1, incarnation="new"
    )
    writes: list = []
    writer = _writer(worker, writes)
    writer._deferred_tasks.append(_task())

    writer._process_deferred_tasks()

    # Nothing is posted over the stale sessions.
    assert writes == [] and worker.sessions_used == []
    # They are dropped, keeping any session already built for "new".
    assert worker.dropped == [(PEER, "new")]
    # The producer frees its blocks; the new consumer frees the ones it allocated.
    assert [ack.transfer_id for ack in worker.moriio_wrapper.done_req_ids] == ["tA"]
    assert worker.moriio_wrapper.sent == [
        ("tA", "10.0.0.1", NOTIFY_PORT + get_port_offset(1, 0, 1), "write_failed")
    ]
    assert worker.moriio_wrapper.done_remote_allocate_req_dict == {}
    assert writer._deferred_tasks == []


def test_write_waits_for_a_new_handshake_then_gives_up_at_the_deferral():
    # The sessions were dropped and no request has handshaken since.
    worker = _producer({})
    worker.moriio_wrapper.done_remote_allocate_req_dict["tA"] = RemoteAllocInfo(
        block_ids=[[2]], decode_dp_rank=0, incarnation="new"
    )
    writes: list = []
    writer = _writer(worker, writes)
    writer._deferred_tasks.append(_task())

    writer._process_deferred_tasks()
    assert writer._deferred_tasks != [] and writes == []
    assert worker.moriio_wrapper.sent == []

    writer._deferred_tasks[0].enqueue_time -= 2 * writer._defer_timeout
    writer._process_deferred_tasks()

    assert writer._deferred_tasks == [] and writes == []
    assert [ack.transfer_id for ack in worker.moriio_wrapper.done_req_ids] == ["tA"]
    assert worker.moriio_wrapper.sent == [
        ("tA", "10.0.0.1", NOTIFY_PORT + get_port_offset(0, 0, 1), "write_failed")
    ]


def test_write_proceeds_once_a_new_handshake_built_sessions():
    sessions: dict[str, str] = {}
    worker = _producer(sessions)
    worker.moriio_wrapper.done_remote_allocate_req_dict["tA"] = RemoteAllocInfo(
        block_ids=[[2]], decode_dp_rank=0, incarnation="new"
    )
    writes: list = []
    writer = _writer(worker, writes)
    writer._deferred_tasks.append(_task())
    writer._process_deferred_tasks()
    assert writes == []

    sessions[f"{PEER}_dp0"] = "new"  # a later request handshook with "new"
    writer._process_deferred_tasks()

    assert writes == [f"{PEER}_dp0"]
    assert worker.moriio_wrapper.sent == []


def test_failed_transfer_tells_the_consumer_and_drops_older_sessions():
    worker = _producer({f"{PEER}_dp0": "inc-1"})
    info = RemoteAllocInfo(block_ids=[[2]], decode_dp_rank=0, incarnation="inc-1")
    worker.moriio_wrapper.done_remote_allocate_req_dict["tA"] = info
    writer = _writer(worker, [])

    writer._handle_write_error(_task(), TransferError("retry exceeded"))

    # Sessions to the incarnation that allocated stay; older ones would go.
    assert worker.dropped == [(PEER, "inc-1")]
    assert worker.moriio_wrapper.sent == [
        ("tA", "10.0.0.1", NOTIFY_PORT + get_port_offset(0, 0, 1), "write_failed")
    ]
    # Reported once even if another layer's task fails too.
    writer._handle_write_error(_task(), TransferError("retry exceeded"))
    assert len(worker.moriio_wrapper.sent) == 1


def test_other_errors_keep_the_old_behavior():
    worker = _producer({f"{PEER}_dp0": "inc-1"})
    worker.moriio_wrapper.done_remote_allocate_req_dict["tA"] = RemoteAllocInfo(
        block_ids=[[2]], decode_dp_rank=0, incarnation="inc-1"
    )
    writer = _writer(worker, [])

    writer._handle_write_error(_task(), KeyError("bug"))

    assert worker.dropped == []
    assert worker.moriio_wrapper.sent == []
    assert [ack.transfer_id for ack in worker.moriio_wrapper.done_req_ids] == ["tA"]


def _meta(incarnation: str, agent: bytes) -> MoRIIOAgentMetadata:
    return MoRIIOAgentMetadata(
        engine_id="decode",
        agent_metadata=agent,
        kv_caches_base_addr=[],
        num_blocks=1,
        block_len=1,
        attn_backend_name="x",
        incarnation=incarnation,
    )


def test_invalidate_remote_peer_drops_only_older_sessions_of_that_peer():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._handshake_lock = threading.Lock()
    deregistered: list[bytes] = []
    worker.moriio_wrapper = SimpleNamespace(
        deregister_remote_engine=deregistered.append, shutdown=lambda: None
    )
    old, new, other = f"{PEER}_dp0", f"{PEER}_dp1_tp2", "10.0.0.9:2_dp0"
    worker.remote_moriio_metadata = {
        old: _meta("old", b"agent-old"),
        new: _meta("new", b"agent-new"),
        other: _meta("old", b"agent-other"),
    }
    worker._remote_agents = {old: {"a"}, new: {"b"}, other: {"c"}}
    worker.layer_name_to_remote_kv_cache_metadata = {old: {}, new: {}, other: {}}
    worker.built_write_session = {old: [1], new: [2], other: [3]}
    worker.write_ready_flags = {PEER: True, "10.0.0.9:2": True}
    worker.load_ready_flag = {PEER: True}

    worker.invalidate_remote_peer(PEER, "test", current_incarnation="new")

    assert set(worker.remote_moriio_metadata) == {new, other}
    assert set(worker._remote_agents) == {new, other}
    assert set(worker.layer_name_to_remote_kv_cache_metadata) == {new, other}
    assert set(worker.built_write_session) == {new, other}
    assert deregistered == [b"agent-old"]
    # The next request to this peer handshakes again before it writes.
    assert worker.write_ready_flags == {"10.0.0.9:2": True}
    assert worker.load_ready_flag == {}
    assert worker.peer_incarnation(old) is None
    assert worker.peer_incarnation(new) == "new"


def test_stale_peer_error_is_a_moriio_error():
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
        MoRIIOError,
    )

    assert issubclass(StalePeerError, MoRIIOError)
