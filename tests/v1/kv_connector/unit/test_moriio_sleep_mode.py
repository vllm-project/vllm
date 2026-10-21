# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO sleep mode with the ``mori`` runtime replaced by fakes."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOHandshakeMetadata,
    RemoteAllocInfo,
    WriteTask,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)

from .utils import make_moriio_writer

pytestmark = pytest.mark.cpu_test


class _FakeWorker(MoRIIOConnectorWorker):
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _agent_metadata(epoch: int = 0) -> MoRIIOAgentMetadata:
    return MoRIIOAgentMetadata(
        engine_id="decode",
        agent_metadata=b"",
        kv_caches_base_addr=[],
        num_blocks=4,
        block_len=64,
        attn_backend_name="TRITON_ATTN",
        registration_epoch=epoch,
    )


def test_release_and_restore_registration():
    ids = iter(range(100))
    wrapper = MagicMock()
    wrapper.register_local_tensor.side_effect = lambda t: msgpack.dumps(next(ids))
    worker = _FakeWorker(
        moriio_wrapper=wrapper,
        model_config=SimpleNamespace(enable_sleep_mode=True),
        _kv_regions={},
        _kv_unregistered=False,
        _registration_epoch=0,
        _agent_metadata=_agent_metadata(),
        _writer=MagicMock(),
        built_write_session={"prefill": ["session"]},
    )
    shared, other = torch.zeros(64), torch.zeros(64)
    served = {
        name: [worker._register_kv_region(t)]
        for name, t in (("l0", shared), ("l1", other), ("l2", shared))
    }
    worker.layer_name_to_local_kv_cache_metadata = served
    lists = dict(served)
    assert served["l0"] == served["l2"] != served["l1"]
    old = [r for (_, r) in worker._kv_regions.values()]

    worker.unregister_kv_caches()
    worker.unregister_kv_caches()
    worker._writer.release_registration.assert_called_once_with()
    assert worker.built_write_session == {}
    assert [c.args[0] for c in wrapper.deregister_local_tensor.call_args_list] == old

    worker.restore_kv_caches()
    worker.restore_kv_caches()
    assert wrapper.register_local_tensor.call_count == 4
    # Updated in place: the handshake listener serves these lists.
    assert all(served[name] is lists[name] for name in lists)
    assert served["l0"] == served["l2"] and not set(old) & {
        *served["l0"],
        *served["l1"],
    }
    assert worker.get_handshake_metadata() == MoRIIOHandshakeMetadata(1)
    assert worker._agent_metadata.registration_epoch == 1


def test_consumer_scheduler_stamps_newest_epoch():
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler._registration_epoch = 0
    scheduler._global_dp_rank = 0
    sock = MagicMock()
    scheduler.paths = {"tcp://127.0.0.1:7000": sock}

    scheduler.set_xfer_handshake_metadata(
        {0: MoRIIOHandshakeMetadata(3), 1: MoRIIOHandshakeMetadata(2)}
    )
    scheduler.send_notify_block("req", "xfer", [1], host="127.0.0.1", port=7000)
    assert msgpack.loads(sock.send.call_args.args[0])["registration_epoch"] == 3


def test_producer_handshakes_again_only_for_a_newer_epoch():
    handshakes = []
    worker = _FakeWorker(
        remote_moriio_metadata={"decode": _agent_metadata(epoch=1)},
        _remote_handshake_args={"decode": ("host", 1, 1, "decode", 0, None)},
        _handshake_initiation_executor=ThreadPoolExecutor(max_workers=1),
        moriio_config=SimpleNamespace(transfer_timeout=5.0),
        built_write_session={"decode": ["stale session"]},
    )

    def handshake(*args):
        # The peer rank serves epoch 2 while its engine stamps 3.
        handshakes.append(args)
        worker.remote_moriio_metadata["decode"] = _agent_metadata(epoch=2)

    worker._moriio_handshake = handshake

    worker._refresh_remote_registration("decode", 1)
    assert not handshakes and worker.built_write_session
    for _ in range(2):
        worker._refresh_remote_registration("decode", 3)
    assert len(handshakes) == 1 and worker.built_write_session == {}
    worker._handshake_initiation_executor.shutdown()


def _write_task(transfer_id: str) -> WriteTask:
    return WriteTask(
        request_id=f"req-{transfer_id}",
        transfer_id=transfer_id,
        dst_engine_id="decode",
        local_block_ids=[[1]],
        remote_block_ids_hint=None,
        layer_name="layer0",
        event=None,
        remote_notify_port=20000,
        remote_ip="10.0.0.1",
    )


def test_writes_scheduled_before_the_release_fail():
    issued, terminal = object(), set[str]()
    wrapper = SimpleNamespace(
        lock=threading.Lock(),
        done_req_ids=[],
        done_remote_allocate_req_dict={
            "tA": RemoteAllocInfo(block_ids=[[7]], transfer_statuses=[issued])
        },
        waited=[],
        notified=[],
        _is_transfer_terminal_locked=terminal.__contains__,
        _mark_transfer_terminal_locked=terminal.add,
    )
    wrapper.waiting_for_transfer_complete = wrapper.waited.append
    wrapper.send_notify = lambda tid, ip, port, message_type: wrapper.notified.append(
        (tid, message_type)
    )
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper, tp_rank=0))
    dequeued = _write_task("tA")
    writer._deferred_tasks.append(_write_task("tB"))

    writer.release_registration()
    # Writes already issued still read the memory being released.
    assert wrapper.waited == [[issued]]
    writer._execute_write_task(dequeued)
    writer._process_deferred_tasks()
    assert wrapper.notified == [("tA", "write_failed"), ("tB", "write_failed")]
    assert terminal == {"tA", "tB"} and writer._deferred_tasks == []

    later = _write_task("tC")
    writer._deferred_tasks.append(later)
    writer._process_deferred_tasks()
    assert writer._deferred_tasks == [later]
