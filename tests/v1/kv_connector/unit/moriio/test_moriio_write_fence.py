# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WRITE-mode consumer fence of the MoRIIO connector (`write_recv_fence`).

A receive that timed out keeps its blocks: the producer's RDMA writes are
one-sided and may still land. With `kv_connector_extra_config["write_recv_fence"]`
the decode workers fence the producer's MoRI connections instead, which makes
the blocks safe to reuse:

  * the scheduler orders one fence per stuck producer, covering every receive
    still in flight from it, since the fence cuts all of them off;
  * each worker fences every TP rank of the producer; only if all connections
    went down does it release the timed-out receives and fail the others, which
    frees their blocks;
  * the worker tells the producer to abort those transfers and unfences it
    once every producer rank acked;
  * the producer drops the transfers, resets its connection and acks;
  * a producer rank whose write failed with nothing in flight tells decode
    with `write_failed`, counted across the ranks writing the same blocks.

They run with MoRI mocked out, without a GPU.
"""

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_common,
    moriio_connector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConnectorMetadata,
    MoRIIOConstants,
    MoRIIOMode,
    MoRIIOTransferAck,
    RemoteAllocInfo,
    TransferError,
    WriteFenceOrder,
    WriteProducer,
    WriteTask,
    get_port_offset,
    moriio_engine_key,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.v1.request import RequestStatus

from ..utils import create_request, make_moriio_writer
from .test_moriio_connector import _setup_kv_transfer_request
from .test_moriio_write_recv_timeout import (
    TIMEOUT_S,
    _Clock,
    _Engine,
    _make_consumer_worker,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

CONSUMER_KEY = "consumer:10.0.0.9:6001:tp0:dp0"
PRODUCER = WriteProducer(
    host="10.0.0.1", handshake_port=6001, notify_port=7000, dp_rank=0, tp_size=2
)
MAPPING = {"xfer-stuck": "stuck", "xfer-coll": "collateral"}
# The producer of `_setup_kv_transfer_request`'s requests, as decode keys it.
TEST_PRODUCER_KEY = moriio_engine_key("producer", "127.0.0.1", 4789, 0, 0)


@pytest.fixture(autouse=True)
def role():
    """The connector checks the process-global role; tests start as consumer."""
    saved = moriio_common.get_role()
    moriio_common.set_role(moriio_common.ROLE.CONSUMER)
    yield moriio_common.set_role
    moriio_common.set_role(saved)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(moriio_connector, "time", fake)
    return fake


class FakeEngine:
    """MoRI IOEngine fence API; every key fences one connection unless told."""

    def __init__(self, results: dict | None = None):
        self.results = results or {}
        self.fenced: list[str] = []
        self.unfenced: list[str] = []

    def fence_remote_engine(self, key: str):
        self.fenced.append(key)
        return self.results.get(key, _fence_result())

    def unfence_remote_engine(self, key: str) -> bool:
        self.unfenced.append(key)
        return True


def _fence_result(ok: bool = True, endpoints_fenced: int = 1, peer_known: bool = True):
    return SimpleNamespace(
        ok=ok and peer_known,
        endpoints_fenced=endpoints_fenced,
        peer_known=peer_known,
        message="x",
    )


# What MoRI returns for a key it never had a connection with.
NEVER_SEEN = _fence_result(endpoints_fenced=0, peer_known=False)


def _recording_wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper()
    wrapper.notify_port = 7100
    wrapper.sent = []

    def send_notify(
        transfer_id, remote_ip, remote_port, message_type=None, message_fields=None
    ):
        wrapper.sent.append(
            (transfer_id, remote_ip, remote_port, message_type, message_fields)
        )

    wrapper.send_notify = send_notify
    return wrapper


def _add_fence_state(worker: MoRIIOConnectorWorker, engine: FakeEngine | None):
    worker.tp_rank = 0
    worker.local_ip = "10.0.0.9"
    worker.engine_key = CONSUMER_KEY
    worker.moriio_engine = engine or FakeEngine()
    worker._unmatched_write_completions = set()
    worker._write_recv_fence = True
    worker._fenced_producer_keys = {}
    worker._fence_pending = {}
    worker._fence_failed_req_ids = set()
    worker._fence_settled_req_ids = set()
    worker._fenced_transfer_ids = OrderedDict()


# --- engine keys ------------------------------------------------------------


def test_producer_engine_key_matches_the_key_the_producer_registers():
    """Decode rebuilds the key; a mismatch would fence nothing."""
    for tp_rank in range(PRODUCER.tp_size):
        assert PRODUCER.engine_key(tp_rank) == moriio_engine_key(
            "producer", "10.0.0.1", 6001, tp_rank, 0
        )
    assert PRODUCER.engine_key(1) == "producer:10.0.0.1:6001:tp1:dp0"


def test_pipeline_stages_have_distinct_engine_keys():
    keys = {moriio_engine_key("producer", "h", 1, 0, 0, pp_rank) for pp_rank in (0, 1)}
    assert keys == {"producer:h:1:tp0:dp0:pp0", "producer:h:1:tp0:dp0:pp1"}


def test_producer_notify_endpoint_matches_the_producer_listener():
    producer = PRODUCER._replace(dp_rank=1)
    assert producer.notify_endpoint(1) == (
        "10.0.0.1",
        7000 + get_port_offset(1, 1),
    )


# --- scheduler: resolving the producer ----------------------------------------


@pytest.fixture
def plain_engine(monkeypatch):
    return _Engine(monkeypatch, 1)


def test_producer_is_resolved_from_the_router_embedded_address(plain_engine):
    scheduler = plain_engine.scheduler.connector.connector_scheduler
    request = create_request(num_tokens=16, do_remote_prefill=True)
    _setup_kv_transfer_request(request, remote_host="10.0.0.7", fake_port=4789)

    producer = scheduler._resolve_write_producer(request, 0, 0)

    assert producer == WriteProducer("10.0.0.7", 4789, 4789, 0, 1)


def test_producer_uses_the_pod_host_of_its_dp_rank(plain_engine):
    scheduler = plain_engine.scheduler.connector.connector_scheduler
    request = create_request(num_tokens=16, do_remote_prefill=True)
    request.kv_transfer_params.update(
        remote_host="10.0.0.1",
        remote_notify_port=7000,
        remote_handshake_port=6001,
        remote_hosts=["10.0.0.1", "10.0.0.2"],
    )

    producer = scheduler._resolve_write_producer(request, 3, 2)

    assert producer.host == "10.0.0.2"
    assert producer.dp_rank == 1


# --- scheduler: ordering fences -------------------------------------------


@pytest.fixture(params=[1, 2], ids=["tp1", "tp2"])
def fence_engine(request, monkeypatch):
    def make_worker():
        worker = _make_consumer_worker()
        _add_fence_state(worker, FakeEngine())
        wrapper = _recording_wrapper()
        wrapper.pop_failed_write_req_ids = (
            worker.moriio_wrapper.pop_failed_write_req_ids
        )
        worker.moriio_wrapper = wrapper
        return worker

    return _Engine(
        monkeypatch,
        request.param,
        extra_config={"write_recv_fence": True},
        make_worker=make_worker,
    )


def _waiting_request(engine, transfer_id: str, host: str = "127.0.0.1"):
    request = create_request(
        num_tokens=4 * engine.scheduler.block_size, do_remote_prefill=True
    )
    _setup_kv_transfer_request(request, remote_host=host, fake_transfer_id=transfer_id)
    engine.scheduler.add_request(request)
    engine.step()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    return request


def test_timeout_orders_a_fence_of_the_producer_covering_its_inflight_recvs(
    fence_engine,
):
    engine = fence_engine
    stuck = _waiting_request(engine, "xfer-stuck")
    engine.clock.now += TIMEOUT_S / 2
    collateral = _waiting_request(engine, "xfer-coll")
    _waiting_request(engine, "xfer-other", host="127.0.0.2")
    engine.clock.now += TIMEOUT_S / 2 + 1

    first = engine.scheduler.schedule()
    second = engine.scheduler.schedule()

    (order,) = first.kv_connector_metadata.write_fence_orders
    assert order.producer.host == "127.0.0.1"
    assert order.requests == {
        stuck.request_id: "xfer-stuck",
        collateral.request_id: "xfer-coll",
    }
    assert not second.kv_connector_metadata.write_fence_orders
    engine.finish(first)
    engine.finish(second)


def test_no_fence_is_ordered_when_the_fence_is_off(plain_engine):
    request = plain_engine.add_waiting_request()
    plain_engine.clock.now += TIMEOUT_S + 1

    out = plain_engine.scheduler.schedule()

    assert out.kv_connector_metadata.write_recv_timed_out_req_ids == {
        request.request_id
    }
    assert not out.kv_connector_metadata.write_fence_orders
    plain_engine.finish(out)


def test_fence_is_off_without_the_receive_timer(monkeypatch):
    engine = _Engine(
        monkeypatch, 1, policy="recompute", extra_config={"write_recv_fence": True}
    )
    assert not engine.scheduler.connector.connector_scheduler._write_recv_fence


# --- through the real scheduler -----------------------------------------------


def test_stuck_producer_is_fenced_and_every_blocked_request_freed(fence_engine):
    engine = fence_engine
    free_before = engine.pool.get_num_free_blocks()
    stuck = _waiting_request(engine, "xfer-stuck")
    engine.clock.now += TIMEOUT_S / 2
    collateral = _waiting_request(engine, "xfer-coll")
    healthy = _waiting_request(engine, "xfer-healthy", host="127.0.0.2")

    engine.clock.now += TIMEOUT_S / 2 + 1
    engine.step()
    assert stuck.status == RequestStatus.FINISHED_ERROR
    assert collateral.status == RequestStatus.FINISHED_ERROR
    engine.step()
    assert stuck.request_id not in engine.scheduler.requests
    assert collateral.request_id not in engine.scheduler.requests

    # The other producer's request is untouched and completes normally.
    assert healthy.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    engine.step(write_done=frozenset({healthy.request_id}))
    engine.step()
    assert healthy.status == RequestStatus.RUNNING
    engine.scheduler.finish_requests(healthy.request_id, RequestStatus.FINISHED_ABORTED)
    engine.step()
    assert engine.pool.get_num_free_blocks() == free_before

    # No timer is left behind.
    engine.clock.now += 10 * TIMEOUT_S
    engine.step()
    connector = engine.scheduler.connector.connector_scheduler
    assert connector._write_recv_deadline == {}
    assert connector._write_recv_producer == {}
    for worker in engine.workers:
        assert worker.moriio_engine.fenced == [TEST_PRODUCER_KEY]


def test_unsafe_fence_keeps_the_blocks_of_its_receives(fence_engine):
    engine = fence_engine
    for worker in engine.workers:
        worker.moriio_engine.results = {TEST_PRODUCER_KEY: _fence_result(ok=False)}
    stuck = _waiting_request(engine, "xfer-stuck")
    engine.clock.now += TIMEOUT_S / 2
    collateral = _waiting_request(engine, "xfer-coll")
    held = engine.pool.get_num_free_blocks()

    engine.clock.now += TIMEOUT_S / 2 + 1
    engine.step()
    engine.step()
    assert stuck.status == RequestStatus.FINISHED_ERROR
    assert stuck.request_id in engine.scheduler.requests
    assert engine.pool.get_num_free_blocks() == held
    # The fence did not fail the other receive; its own timer does, without a
    # second fence.
    assert collateral.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    engine.clock.now += TIMEOUT_S / 2
    engine.step()
    assert collateral.status == RequestStatus.FINISHED_ERROR
    assert engine.pool.get_num_free_blocks() == held
    for worker in engine.workers:
        assert worker.moriio_engine.fenced == [TEST_PRODUCER_KEY]

    # The write_done of a receive the timeout failed still frees its blocks.
    engine.step(write_done=frozenset({stuck.request_id}))
    assert stuck.request_id not in engine.scheduler.requests


def test_fence_unsafe_on_one_rank_keeps_the_blocks(fence_engine):
    engine = fence_engine
    if len(engine.workers) < 2:
        pytest.skip("needs more than one TP rank")
    engine.workers[1].moriio_engine.results = {
        TEST_PRODUCER_KEY: _fence_result(ok=False)
    }
    free_before = engine.pool.get_num_free_blocks()
    stuck = _waiting_request(engine, "xfer-stuck")
    held = engine.pool.get_num_free_blocks()

    engine.clock.now += TIMEOUT_S + 1
    engine.step()
    engine.step()
    assert stuck.status == RequestStatus.FINISHED_ERROR
    assert engine.pool.get_num_free_blocks() == held

    # The rank that could not fence frees them once its write is over.
    engine.step(write_done=frozenset({stuck.request_id}), ranks={1})
    assert stuck.request_id not in engine.scheduler.requests
    assert engine.pool.get_num_free_blocks() == free_before


# --- worker: fencing ---------------------------------------------------------


def _fence_worker(engine: FakeEngine | None = None) -> MoRIIOConnectorWorker:
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.WRITE
    worker.moriio_wrapper = _recording_wrapper()
    worker.transfer_id_to_request_id = {}
    worker._unmatched_write_failures = OrderedDict()
    worker._write_recv_timed_out = set()
    worker._write_recv_reported = set()
    worker._write_recv_unreleased = set()
    _add_fence_state(worker, engine)
    return worker


def _order(fence_id: str = "f1", producer: WriteProducer = PRODUCER, **requests):
    return WriteFenceOrder(
        fence_id=fence_id,
        producer=producer,
        requests=requests or {"stuck": "xfer-stuck", "collateral": "xfer-coll"},
    )


def _apply(worker, *orders, mapping: dict | None = None, timed_out=()):
    metadata = MoRIIOConnectorMetadata()
    metadata.transfer_id_to_request_id = MAPPING if mapping is None else mapping
    metadata.write_recv_timed_out_req_ids = set(timed_out)
    metadata.write_fence_orders = list(orders)
    worker.start_load_kv(metadata)


def test_safe_fence_fails_every_recv_of_the_order_once():
    worker = _fence_worker()

    _apply(worker, _order())
    results = worker.get_transfer_results()

    assert worker.moriio_engine.fenced == [
        PRODUCER.engine_key(0),
        PRODUCER.engine_key(1),
    ]
    assert results.failed_recving == {"stuck", "collateral"}
    assert results.finished_recving == {"stuck", "collateral"}
    assert results.finished_sending == set()
    # Reported once: a second report would reach the scheduler after it freed
    # the request.
    _apply(worker)
    again = worker.get_transfer_results()
    assert again.finished_recving == set()
    assert again.failed_recving == set()


def test_fence_releases_a_receive_the_timeout_failed():
    """The scheduler holds a timed-out receive's blocks until a release."""
    worker = _fence_worker()
    _apply(worker, timed_out={"stuck"})
    assert worker.get_transfer_results().failed_recving == {"stuck"}

    _apply(worker, _order())
    results = worker.get_transfer_results()

    assert results.finished_sending == {"stuck"}
    assert results.finished_recving == {"collateral"}
    assert results.failed_recving == {"collateral"}


def test_fence_with_the_timeout_fails_and_releases_in_one_step():
    worker = _fence_worker()

    _apply(worker, _order(), timed_out={"stuck"})
    results = worker.get_transfer_results()

    assert results.finished_recving == {"stuck", "collateral"}
    assert results.failed_recving == {"stuck", "collateral"}
    assert results.finished_sending == {"stuck"}


def test_fence_tells_every_producer_rank_to_abort():
    worker = _fence_worker()

    _apply(worker, _order())

    sent = worker.moriio_wrapper.sent
    assert [(s[0], s[1], s[2], s[3]) for s in sent] == [
        ("f1", "10.0.0.1", 7000 + get_port_offset(0, tp), "fence_abort")
        for tp in range(2)
    ]
    assert sent[0][4] == {
        "transfer_ids": ["xfer-stuck", "xfer-coll"],
        "consumer_key": CONSUMER_KEY,
        "reply_host": "10.0.0.9",
        "reply_port": 7100,
    }


@pytest.mark.parametrize(
    "result",
    [_fence_result(ok=False), NEVER_SEEN],
    ids=["a-connection-could-not-be-fenced", "no-key-ever-seen"],
)
def test_unsafe_fence_reports_nothing(result):
    engine = FakeEngine(
        {PRODUCER.engine_key(0): result, PRODUCER.engine_key(1): result}
    )
    worker = _fence_worker(engine)

    _apply(worker, _order(), timed_out={"stuck"})
    results = worker.get_transfer_results()

    # Only the timeout's own report: failed, with its blocks held.
    assert results.finished_recving == {"stuck"}
    assert results.finished_sending == set()
    # The producer is still told to stop.
    assert len(worker.moriio_wrapper.sent) == 2


def test_fence_without_connections_on_one_rank_is_safe():
    """With equal TP each decode rank only holds a connection to one producer
    rank; the fence of the other key finds nothing and that is fine."""
    engine = FakeEngine({PRODUCER.engine_key(1): NEVER_SEEN})
    worker = _fence_worker(engine)

    _apply(worker, _order())

    assert worker.get_transfer_results().failed_recving == {"stuck", "collateral"}


def test_a_known_producer_whose_connections_are_gone_is_safe_to_fence():
    """A second fence finds nothing left to cut, as the first one cut it all;
    the producer is still known, so its receives may be freed."""
    engine = FakeEngine(
        {
            PRODUCER.engine_key(0): _fence_result(endpoints_fenced=0),
            PRODUCER.engine_key(1): NEVER_SEEN,
        }
    )
    worker = _fence_worker(engine)

    _apply(worker, _order())

    assert worker.get_transfer_results().failed_recving == {"stuck", "collateral"}


@pytest.mark.parametrize(
    "result",
    # MoRI before peer_known reported such a key ok, with nothing fenced.
    [NEVER_SEEN, SimpleNamespace(ok=True, endpoints_fenced=0, message="")],
    ids=["mori-says-unknown", "older-mori"],
)
def test_a_key_never_seen_is_never_taken_for_fenced(result):
    """Keys that match no engine of the producer stay unsafe on every fence,
    not only the first: the real producer is not fenced and may still write."""
    engine = FakeEngine({PRODUCER.engine_key(tp): result for tp in range(2)})
    worker = _fence_worker(engine)
    mapping = {**MAPPING, "xfer-late": "late"}

    _apply(worker, _order("f1"), mapping=mapping)
    _apply(worker, _order("f2", late="xfer-late"), mapping=mapping)
    results = worker.get_transfer_results()

    assert results.failed_recving == set()
    assert results.finished_recving == set()


def test_write_done_that_beat_the_fence_is_reported_as_success():
    worker = _fence_worker()
    worker.moriio_wrapper._handle_structured_message(
        {"type": "write_done", "transfer_id": "xfer-stuck"}
    )

    _apply(worker, _order())
    results = worker.get_transfer_results()

    assert results.finished_recving == {"stuck", "collateral"}
    assert results.failed_recving == {"collateral"}


def test_write_done_after_the_fence_is_not_reported_again():
    worker = _fence_worker()
    _apply(worker, _order())
    worker.get_transfer_results()

    # The scheduler still maps the requests when the write_done lands.
    worker.moriio_wrapper._handle_structured_message(
        {"type": "write_done", "transfer_id": "xfer-coll"}
    )
    _apply(worker)
    results = worker.get_transfer_results()

    assert results.finished_recving == set()
    assert results.finished_sending == set()


def test_late_write_done_of_a_fenced_transfer_is_dropped():
    worker = _fence_worker()
    _apply(worker, _order())
    worker.get_transfer_results()

    # The request is gone, so its transfer id no longer maps to it.
    worker.moriio_wrapper._handle_structured_message(
        {"type": "write_done", "transfer_id": "xfer-stuck"}
    )
    _apply(worker, mapping={})
    results = worker.get_transfer_results()

    assert results.finished_recving == set()
    assert worker._unmatched_write_completions == set()


def test_fence_does_not_report_a_request_already_gone():
    worker = _fence_worker()

    _apply(worker, _order(), mapping={"xfer-coll": "collateral"})

    assert worker.get_transfer_results().finished_recving == {"collateral"}


# --- worker: acks and unfence -------------------------------------------------


def _ack(worker, fence_id: str, tp_rank: int, producer=PRODUCER):
    worker.moriio_wrapper._handle_structured_message(
        {
            "type": "fence_ack",
            "transfer_id": fence_id,
            "producer_key": producer.engine_key(tp_rank),
        }
    )


def test_producer_is_unfenced_once_every_rank_acked():
    worker = _fence_worker()
    _apply(worker, _order())
    worker.get_transfer_results()

    _ack(worker, "f1", 0)
    worker.get_transfer_results()
    assert worker.moriio_engine.unfenced == []

    _ack(worker, "f1", 1)
    worker.get_transfer_results()
    assert sorted(worker.moriio_engine.unfenced) == [
        PRODUCER.engine_key(0),
        PRODUCER.engine_key(1),
    ]
    assert worker._fence_pending == {}
    assert worker._fenced_producer_keys == {}


def test_producer_stays_fenced_until_every_fence_of_it_is_acked():
    worker = _fence_worker()
    _apply(worker, _order("f1"))
    _apply(worker, _order("f2", late="xfer-late"))

    for tp in range(2):
        _ack(worker, "f1", tp)
    worker.get_transfer_results()
    assert worker.moriio_engine.unfenced == []

    for tp in range(2):
        _ack(worker, "f2", tp)
    worker.get_transfer_results()
    assert len(worker.moriio_engine.unfenced) == 2


def test_unacked_abort_is_resent(clock):
    worker = _fence_worker()
    _apply(worker, _order())
    _ack(worker, "f1", 0)
    worker.get_transfer_results()
    assert len(worker.moriio_wrapper.sent) == 2

    clock.now += MoRIIOConstants.FENCE_ABORT_RESEND_INTERVAL + 1
    worker.get_transfer_results()

    resent = worker.moriio_wrapper.sent[2:]
    assert [(s[2], s[3]) for s in resent] == [
        (7000 + get_port_offset(0, 1), "fence_abort")
    ]


# --- wrapper: write_failed across writers ----------------------------------


def _writer_message(kind: str, sender: list[int]) -> dict:
    return {"type": kind, "transfer_id": "t", "num_writers": 2, "sender": sender}


def test_failure_from_one_writer_waits_for_the_other_writers():
    wrapper = MoRIIOWrapper()

    wrapper._handle_structured_message(_writer_message("write_failed", [0, 0]))
    assert wrapper.pop_failed_write_req_ids() == set()

    wrapper._handle_structured_message(_writer_message("write_done", [0, 1]))
    assert wrapper.pop_failed_write_req_ids() == {"t"}
    assert wrapper.pop_finished_write_req_ids() == set()


def test_a_writer_reporting_twice_counts_once():
    wrapper = MoRIIOWrapper()

    wrapper._handle_structured_message(_writer_message("write_failed", [0, 0]))
    wrapper._handle_structured_message(_writer_message("write_failed", [0, 0]))

    assert wrapper.pop_failed_write_req_ids() == set()


def test_failure_without_writer_count_is_immediate():
    """Sent when the allocation never arrived, so nothing was written."""
    wrapper = MoRIIOWrapper()
    wrapper._handle_structured_message(_writer_message("write_done", [0, 0]))

    wrapper._handle_structured_message({"type": "write_failed", "transfer_id": "t"})

    assert wrapper.pop_failed_write_req_ids() == {"t"}
    assert wrapper._write_done_senders == {}


def test_fence_aborts_are_popped_once(role):
    role(moriio_common.ROLE.PRODUCER)
    wrapper = MoRIIOWrapper()
    wrapper._handle_structured_message({"type": "fence_abort", "transfer_id": "f1"})

    assert wrapper.pop_fence_aborts() == [{"type": "fence_abort", "transfer_id": "f1"}]
    assert wrapper.pop_fence_aborts() == []


# --- producer: abort, failure and finalize --------------------------------


class _Status:
    def __init__(self, state: str):
        self.state = state

    def Succeeded(self):
        return self.state == "ok"

    def Failed(self):
        return self.state == "failed"


def _producer_writer(engine: FakeEngine | None = None, pp_rank=0, pp_size=1):
    wrapper = _recording_wrapper()
    worker = SimpleNamespace(
        moriio_wrapper=wrapper,
        moriio_engine=engine or FakeEngine(),
        engine_key=PRODUCER.engine_key(1),
        tp_rank=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
    )
    return make_moriio_writer(worker), wrapper


def _alloc(wrapper, transfer_id: str, *statuses: str) -> RemoteAllocInfo:
    info = RemoteAllocInfo(block_ids=[[4, 5]], completion_request_id="req")
    info.completion_remote_ip = "10.0.0.9"
    info.completion_notify_port = 7100
    info.transfer_statuses.extend(_Status(s) for s in statuses)
    wrapper.done_remote_allocate_req_dict[transfer_id] = info
    return info


def test_abort_drops_the_transfers_resets_the_connection_and_acks(role):
    role(moriio_common.ROLE.PRODUCER)
    writer, wrapper = _producer_writer()
    _alloc(wrapper, "t1", "ok")
    writer._scheduled_writes["t1"] = 3
    wrapper._handle_structured_message(
        {
            "type": "fence_abort",
            "transfer_id": "f1",
            "transfer_ids": ["t1", "t-unknown"],
            "consumer_key": CONSUMER_KEY,
            "reply_host": "10.0.0.9",
            "reply_port": 7100,
        }
    )

    writer._process_fence_aborts()

    assert wrapper._is_transfer_terminal_locked("t1")
    assert wrapper._is_transfer_terminal_locked("t-unknown")
    assert MoRIIOTransferAck("t1") in wrapper.done_req_ids
    assert "t1" not in wrapper.done_remote_allocate_req_dict
    assert "t1" not in writer._scheduled_writes
    engine = writer.worker.moriio_engine
    assert engine.fenced == [CONSUMER_KEY]
    assert engine.unfenced == [CONSUMER_KEY]
    assert wrapper.sent == [
        (
            "f1",
            "10.0.0.9",
            7100,
            "fence_ack",
            {"producer_key": PRODUCER.engine_key(1)},
        )
    ]
    # Tasks of an aborted transfer are never posted.
    assert not writer.schedule_write(
        WriteTask(
            request_id="req",
            transfer_id="t1",
            dst_engine_id="d",
            local_block_ids=[[1]],
            remote_block_ids_hint=None,
            layer_name="layer0",
            event=None,
            remote_notify_port=7100,
            remote_ip="10.0.0.9",
        )
    )


def test_resent_abort_is_acked_again_without_another_reset(role):
    role(moriio_common.ROLE.PRODUCER)
    writer, wrapper = _producer_writer()
    abort = {
        "type": "fence_abort",
        "transfer_id": "f1",
        "transfer_ids": [],
        "consumer_key": CONSUMER_KEY,
        "reply_host": "10.0.0.9",
        "reply_port": 7100,
    }
    wrapper._handle_structured_message(dict(abort))
    writer._process_fence_aborts()
    wrapper._handle_structured_message(dict(abort))
    writer._process_fence_aborts()

    engine = writer.worker.moriio_engine
    assert engine.fenced == [CONSUMER_KEY]
    assert [m[3] for m in wrapper.sent] == ["fence_ack", "fence_ack"]


def test_failed_transfer_with_nothing_in_flight_tells_decode():
    writer, wrapper = _producer_writer()
    _alloc(wrapper, "t1", "ok", "failed")

    writer._fail_transfer("t1")

    assert wrapper.sent == [
        ("t1", "10.0.0.9", 7100, "write_failed", {"num_writers": 1, "sender": [0, 1]})
    ]
    assert wrapper.done_req_ids == [MoRIIOTransferAck("t1")]


def test_failed_transfer_counts_every_pipeline_stage_as_a_writer():
    """Decode settles a transfer once every stage reported, as for write_done."""
    writer, wrapper = _producer_writer(pp_rank=1, pp_size=2)
    _alloc(wrapper, "t1", "failed")

    writer._fail_transfer("t1")

    assert wrapper.sent[0][4] == {"num_writers": 2, "sender": [1, 1]}


def test_failed_transfer_with_writes_in_flight_leaves_decode_to_its_timer():
    writer, wrapper = _producer_writer()
    _alloc(wrapper, "t1", "ok", "pending")

    writer._fail_transfer("t1")

    assert wrapper.sent == []
    # Prefill's own blocks are released either way.
    assert wrapper.done_req_ids == [MoRIIOTransferAck("t1")]


def test_failed_transfer_that_was_aborted_sends_nothing():
    writer, wrapper = _producer_writer()
    _alloc(wrapper, "t1", "failed")
    with wrapper.lock:
        wrapper._mark_transfer_terminal_locked("t1")

    writer._fail_transfer("t1")

    assert wrapper.sent == []


def test_failed_wait_fails_the_transfer_instead_of_raising():
    """Finalize can run on the model runner thread."""
    writer, wrapper = _producer_writer()
    info = _alloc(wrapper, "t1", "failed")
    info.writes_expected = 1
    info.writes_done = 1

    def fail(statuses):
        raise TransferError("lost")

    wrapper.waiting_for_transfer_complete = fail

    writer._finalize_if_complete("t1", info)

    assert wrapper.sent == [
        ("t1", "10.0.0.9", 7100, "write_failed", {"num_writers": 1, "sender": [0, 1]})
    ]


def test_finalize_after_an_abort_sends_no_write_done():
    writer, wrapper = _producer_writer()
    info = _alloc(wrapper, "t1", "ok")
    info.writes_expected = 1
    info.writes_done = 1

    def abort_during_wait(statuses):
        with wrapper.lock:
            wrapper._mark_transfer_terminal_locked("t1")

    wrapper.waiting_for_transfer_complete = abort_during_wait

    writer._finalize_if_complete("t1", info)

    assert wrapper.sent == []
