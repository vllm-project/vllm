# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WRITE-mode consumer fence of the MoRIIO connector (write_recv_fence).

When a receive times out, decode cannot free its blocks on the timeout alone:
the producer's RDMA writes are one-sided and may still land. With
``kv_connector_extra_config["write_recv_fence"]`` the decode workers fence the
producer's MoRI connections instead, which makes the blocks safe to reuse:

  * the scheduler orders one fence per stuck producer, covering every receive
    still in flight from it, since the fence cuts all of them off;
  * each worker fences every TP rank of the producer and reports the receives
    as failed (freeing their blocks) only if all connections went down;
  * the worker tells the producer to abort those transfers and unfences it
    once every producer rank acked;
  * the producer drops the transfers, resets its connection and acks;
  * a producer rank whose write failed with nothing in flight tells decode
    with ``write_failed``, counted across the ranks writing the same blocks.

They run with MoRI mocked out, without a GPU.
"""

import time
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

from .test_moriio_connector import _setup_kv_transfer_request
from .test_moriio_connector import create_vllm_config as create_moriio_config
from .test_moriio_write_recv_timeout import (
    _Clock,
    _make_blocks,
    _make_request,
    _make_scheduler,
)
from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    make_moriio_writer,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

TIMEOUT_S = 60.0
CONSUMER_KEY = "consumer:10.0.0.9:6001:tp0:dp0"
PRODUCER = WriteProducer(
    host="10.0.0.1", handshake_port=6001, notify_port=7000, dp_rank=0, tp_size=2
)


@pytest.fixture
def role():
    """The wrapper's message handlers check the process-global role."""
    saved = moriio_common.get_role()
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
        return self.results.get(
            key, SimpleNamespace(ok=True, endpoints_fenced=1, message="")
        )

    def unfence_remote_engine(self, key: str) -> bool:
        self.unfenced.append(key)
        return True


def _fence_result(ok: bool = True, endpoints_fenced: int = 1):
    return SimpleNamespace(ok=ok, endpoints_fenced=endpoints_fenced, message="x")


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


# --- engine keys ------------------------------------------------------------


def test_producer_engine_key_matches_the_key_the_producer_registers():
    """Decode rebuilds the key; a mismatch would fence nothing."""
    for tp_rank in range(PRODUCER.tp_size):
        assert PRODUCER.engine_key(tp_rank) == moriio_engine_key(
            "producer", "10.0.0.1", 6001, tp_rank, 0
        )
    assert PRODUCER.engine_key(1) == "producer:10.0.0.1:6001:tp1:dp0"


def test_producer_notify_endpoint_matches_the_producer_listener():
    producer = PRODUCER._replace(dp_rank=1)
    assert producer.notify_endpoint(1) == (
        "10.0.0.1",
        7000 + get_port_offset(1, 1),
    )


def test_producer_is_resolved_from_the_router_embedded_address():
    scheduler = _make_scheduler()
    request = create_request(num_tokens=16, do_remote_prefill=True)
    _setup_kv_transfer_request(request, remote_host="10.0.0.7", fake_port=4789)

    producer = scheduler._resolve_write_producer(request, 0, 0)

    assert producer == WriteProducer("10.0.0.7", 4789, 4789, 0, 1)


def test_producer_uses_the_pod_host_of_its_dp_rank():
    scheduler = _make_scheduler()
    request = _make_request()
    request.kv_transfer_params.update(
        remote_handshake_port=6001, remote_hosts=["10.0.0.1", "10.0.0.2"]
    )

    producer = scheduler._resolve_write_producer(request, 3, 2)

    assert producer.host == "10.0.0.2"
    assert producer.dp_rank == 1


def test_unaddressable_producer_still_raises_on_notify():
    scheduler = _make_scheduler()
    request = _make_request()
    del request.kv_transfer_params["remote_host"]

    with pytest.raises(ValueError, match="remote_host"):
        scheduler.update_state_after_alloc(request, _make_blocks([10]), 16)


# --- scheduler: ordering fences -------------------------------------------


def _fence_scheduler() -> moriio_connector.MoRIIOConnectorScheduler:
    scheduler = _make_scheduler()
    scheduler._write_recv_fence = True
    scheduler.engine_id = "decode"
    return scheduler


def _arm_from(scheduler, request_id: str, producer: WriteProducer, blocks):
    scheduler._arm_write_recv_timer(
        request_id, _make_blocks(blocks), 16, producer, f"xfer-{request_id}"
    )


def test_timeout_orders_a_fence_of_the_producer_covering_its_inflight_recvs():
    scheduler = _fence_scheduler()
    other = PRODUCER._replace(host="10.0.0.2")
    _arm_from(scheduler, "stuck", PRODUCER, [10])
    _arm_from(scheduler, "collateral", PRODUCER, [11])
    _arm_from(scheduler, "elsewhere", other, [12])
    scheduler._write_recv_deadline["stuck"] = time.monotonic() - 1

    meta = scheduler.build_connector_meta(SimpleNamespace())

    (order,) = meta.write_fence_orders
    assert order.producer == PRODUCER
    assert order.requests == {"stuck": "xfer-stuck", "collateral": "xfer-collateral"}
    assert order.fence_id == "decode-fence1"
    # The timed-out request is still failed through its blocks.
    assert meta.write_load_failed_block_ids == [10]
    # The collateral request keeps its timer in case a fence is unsafe.
    assert "collateral" in scheduler._write_recv_deadline
    assert "elsewhere" in scheduler._write_recv_producer
    # Shipped once.
    assert scheduler.build_connector_meta(SimpleNamespace()).write_fence_orders == []


def test_a_fenced_recv_is_not_fenced_again_when_its_timer_runs_out():
    scheduler = _fence_scheduler()
    _arm_from(scheduler, "stuck", PRODUCER, [10])
    _arm_from(scheduler, "collateral", PRODUCER, [11])
    scheduler._write_recv_deadline["stuck"] = time.monotonic() - 1
    scheduler.build_connector_meta(SimpleNamespace())

    scheduler._write_recv_deadline["collateral"] = time.monotonic() - 1
    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.write_fence_orders == []
    assert meta.write_load_failed_block_ids == [11]


def test_orphaned_recv_is_fenced_so_its_blocks_come_back():
    scheduler = _fence_scheduler()
    _arm_from(scheduler, "orphan", PRODUCER, [10])
    scheduler._write_recv_orphaned.add("orphan")
    scheduler._write_recv_deadline["orphan"] = time.monotonic() - 1

    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.write_fence_orders[0].requests == {"orphan": "xfer-orphan"}
    assert meta.write_load_failed_block_ids == []


def test_no_fence_is_ordered_when_the_fence_is_off():
    scheduler = _make_scheduler()
    _arm_from(scheduler, "stuck", PRODUCER, [10])
    scheduler._write_recv_deadline["stuck"] = time.monotonic() - 1

    meta = scheduler.build_connector_meta(SimpleNamespace())

    assert meta.write_fence_orders == []
    assert meta.write_load_failed_block_ids == [10]
    assert scheduler._write_recv_producer == {}


def test_finished_recving_forgets_the_producer():
    scheduler = _fence_scheduler()
    _arm_from(scheduler, "req", PRODUCER, [10])

    scheduler.update_connector_output(
        SimpleNamespace(finished_recving={"req"}, finished_sending=None)
    )

    assert scheduler._write_recv_producer == {}
    assert scheduler._write_recv_transfer_id == {}


# --- worker: fencing ---------------------------------------------------------


def _fence_worker(engine: FakeEngine | None = None) -> MoRIIOConnectorWorker:
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.WRITE
    worker.tp_rank = 0
    worker.local_ip = "10.0.0.9"
    worker.engine_key = CONSUMER_KEY
    worker.moriio_engine = engine or FakeEngine()
    worker.moriio_wrapper = _recording_wrapper()
    worker.transfer_id_to_request_id = {}
    worker._invalid_block_ids = set()
    worker._unmatched_write_completions = set()
    worker._unmatched_write_failures = OrderedDict()
    worker._write_recv_fence = True
    worker._fenced_producer_keys = {}
    worker._fence_pending = {}
    worker._fence_failed_req_ids = set()
    worker._fenced_transfer_ids = OrderedDict()
    worker._reported_recv_req_ids = OrderedDict()
    return worker


def _order(fence_id: str = "f1", producer: WriteProducer = PRODUCER, **requests):
    return WriteFenceOrder(
        fence_id=fence_id,
        producer=producer,
        requests=requests or {"stuck": "xfer-stuck", "collateral": "xfer-coll"},
    )


def _apply(worker, *orders, mapping: dict | None = None):
    metadata = MoRIIOConnectorMetadata()
    metadata.transfer_id_to_request_id = mapping or {}
    metadata.write_fence_orders = list(orders)
    worker.start_load_kv(metadata)


def test_safe_fence_fails_every_recv_of_the_order_once(role):
    role(moriio_common.ROLE.CONSUMER)
    worker = _fence_worker()

    _apply(worker, _order())
    results = worker.get_transfer_results()

    assert worker.moriio_engine.fenced == [
        PRODUCER.engine_key(0),
        PRODUCER.engine_key(1),
    ]
    assert results.failed_recving == {"stuck", "collateral"}
    assert results.finished_recving == {"stuck", "collateral"}
    # Reported once: a second report would reach the scheduler after it freed
    # the request.
    again = worker.get_transfer_results()
    assert again.finished_recving == set()
    assert again.failed_recving == set()


def test_fence_tells_every_producer_rank_to_abort(role):
    role(moriio_common.ROLE.CONSUMER)
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
    [_fence_result(ok=False), _fence_result(endpoints_fenced=0)],
    ids=["a-connection-could-not-be-fenced", "no-connection-found"],
)
def test_unsafe_fence_keeps_the_blocks_allocated(role, result):
    role(moriio_common.ROLE.CONSUMER)
    engine = FakeEngine(
        {PRODUCER.engine_key(0): result, PRODUCER.engine_key(1): result}
    )
    worker = _fence_worker(engine)

    _apply(worker, _order())
    results = worker.get_transfer_results()

    assert results.finished_recving == set()
    assert results.failed_recving == set()
    # The producer is still told to stop.
    assert len(worker.moriio_wrapper.sent) == 2


def test_fence_without_connections_on_one_rank_is_safe(role):
    """With equal TP each decode rank only holds a connection to one producer
    rank; the fence of the other key finds nothing and that is fine."""
    role(moriio_common.ROLE.CONSUMER)
    engine = FakeEngine({PRODUCER.engine_key(1): _fence_result(endpoints_fenced=0)})
    worker = _fence_worker(engine)

    _apply(worker, _order())

    assert worker.get_transfer_results().failed_recving == {"stuck", "collateral"}


def test_write_done_that_beat_the_fence_is_reported_as_success(role):
    role(moriio_common.ROLE.CONSUMER)
    worker = _fence_worker()
    worker.moriio_wrapper._handle_structured_message(
        {"type": "write_done", "transfer_id": "xfer-stuck"}
    )

    _apply(worker, _order(), mapping={"xfer-stuck": "stuck", "xfer-coll": "collateral"})
    results = worker.get_transfer_results()

    assert results.finished_recving == {"stuck", "collateral"}
    assert results.failed_recving == {"collateral"}


def test_late_write_done_of_a_fenced_transfer_is_dropped(role):
    role(moriio_common.ROLE.CONSUMER)
    worker = _fence_worker()
    _apply(worker, _order())
    worker.get_transfer_results()

    # The request is gone, so its transfer id no longer maps to it.
    worker.moriio_wrapper._handle_structured_message(
        {"type": "write_done", "transfer_id": "xfer-stuck"}
    )
    results = worker.get_transfer_results()

    assert results.finished_recving == set()
    assert worker._unmatched_write_completions == set()


# --- worker: acks and unfence -------------------------------------------------


def _ack(worker, fence_id: str, tp_rank: int, producer=PRODUCER):
    worker.moriio_wrapper._handle_structured_message(
        {
            "type": "fence_ack",
            "transfer_id": fence_id,
            "producer_key": producer.engine_key(tp_rank),
        }
    )


def test_producer_is_unfenced_once_every_rank_acked(role):
    role(moriio_common.ROLE.CONSUMER)
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


def test_producer_stays_fenced_until_every_fence_of_it_is_acked(role):
    role(moriio_common.ROLE.CONSUMER)
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


def test_unacked_abort_is_resent(role, clock):
    role(moriio_common.ROLE.CONSUMER)
    worker = _fence_worker()
    _apply(worker, _order())
    _ack(worker, "f1", 0)
    worker.get_transfer_results()
    assert len(worker.moriio_wrapper.sent) == 2

    clock.advance(MoRIIOConstants.FENCE_ABORT_RESEND_INTERVAL + 1)
    worker.get_transfer_results()

    resent = worker.moriio_wrapper.sent[2:]
    assert [(s[2], s[3]) for s in resent] == [
        (7000 + get_port_offset(0, 1), "fence_abort")
    ]


# --- wrapper: write_failed across writers ----------------------------------


def test_failure_from_one_writer_waits_for_the_other_writers(role):
    role(moriio_common.ROLE.CONSUMER)
    wrapper = MoRIIOWrapper()

    wrapper._handle_structured_message(
        {"type": "write_failed", "transfer_id": "t", "num_writers": 2}
    )
    assert wrapper.pop_failed_write_req_ids() == set()

    wrapper._handle_structured_message(
        {"type": "write_done", "transfer_id": "t", "num_writers": 2}
    )
    assert wrapper.pop_failed_write_req_ids() == {"t"}
    assert wrapper.pop_finished_write_req_ids() == set()


def test_failure_without_writer_count_is_immediate(role):
    """Sent when the allocation never arrived, so nothing was written."""
    role(moriio_common.ROLE.CONSUMER)
    wrapper = MoRIIOWrapper()
    wrapper._handle_structured_message(
        {"type": "write_done", "transfer_id": "t", "num_writers": 2}
    )

    wrapper._handle_structured_message({"type": "write_failed", "transfer_id": "t"})

    assert wrapper.pop_failed_write_req_ids() == {"t"}
    assert wrapper._write_done_counts == {}


def test_fence_aborts_are_held_back_by_the_debug_delay(role):
    role(moriio_common.ROLE.PRODUCER)
    wrapper = MoRIIOWrapper()
    wrapper._handle_structured_message({"type": "fence_abort", "transfer_id": "f1"})

    assert wrapper.pop_fence_aborts(min_age=60) == []
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


def _producer_writer(engine: FakeEngine | None = None):
    wrapper = _recording_wrapper()
    worker = SimpleNamespace(
        moriio_wrapper=wrapper,
        moriio_engine=engine or FakeEngine(),
        engine_key=PRODUCER.engine_key(1),
        tp_rank=1,
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


def test_resent_abort_is_reacked_without_another_reset(role):
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
        ("t1", "10.0.0.9", 7100, "write_failed", {"num_writers": 1})
    ]
    assert wrapper.done_req_ids == [MoRIIOTransferAck("t1")]


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
        ("t1", "10.0.0.9", 7100, "write_failed", {"num_writers": 1})
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


def test_debug_hold_holds_only_the_nth_transfer():
    writer, _ = _producer_writer()
    writer.ensure_worker_started = lambda: None
    writer._debug_hold_nth = 2
    writer._debug_hold_s = 30.0
    writer._debug_hold_until = {}

    def task(transfer_id, layer):
        return WriteTask(
            request_id=f"req-{transfer_id}",
            transfer_id=transfer_id,
            dst_engine_id="d",
            local_block_ids=[[1]],
            remote_block_ids_hint=None,
            layer_name=layer,
            event=None,
            remote_notify_port=7100,
            remote_ip="10.0.0.9",
        )

    for transfer_id in ("t1", "t2", "t3"):
        for layer in ("layer0", "layer1"):
            writer.schedule_write(task(transfer_id, layer))

    assert list(writer._debug_hold_until) == ["t2"]
    assert writer._is_held(task("t2", "layer0"))
    assert not writer._is_held(task("t1", "layer0"))


# --- through the real scheduler -----------------------------------------------
#
# The real vLLM Scheduler and block pool, the real connector scheduler and the
# real worker methods, with MoRI's engine mocked: a stuck producer is fenced,
# its receives fail, and their blocks return to the pool.


@pytest.fixture
def fenced_consumer(monkeypatch, clock):
    config = create_moriio_config(role="kv_consumer")
    extra = config.kv_transfer_config.kv_connector_extra_config
    extra["write_recv_timeout"] = TIMEOUT_S
    extra["write_recv_fence"] = True
    scheduler = create_scheduler(config)
    connector = scheduler.connector.connector_scheduler
    monkeypatch.setattr(connector, "send_notify_block", lambda **kwargs: None)
    worker = _fence_worker()
    return scheduler, connector, worker


def _step(scheduler, worker, finished_recving: set[str] | None = None):
    out = scheduler.schedule()
    worker.start_load_kv(out.kv_connector_metadata)
    invalid = worker.get_block_ids_with_load_errors()
    results = worker.get_transfer_results()
    finished = (finished_recving or set()) | results.finished_recving
    scheduled = [scheduler.requests[i] for i in out.num_scheduled_tokens]
    runner_output = create_model_runner_output(
        scheduled, finished_recving=finished, invalid_block_ids=invalid
    )
    runner_output.kv_connector_output.failed_recving = results.failed_recving
    scheduler.update_from_output(out, runner_output)
    return results


def _waiting_request(scheduler, worker, transfer_id: str, host: str = "127.0.0.1"):
    request = create_request(
        num_tokens=4 * scheduler.block_size, do_remote_prefill=True
    )
    _setup_kv_transfer_request(request, remote_host=host, fake_transfer_id=transfer_id)
    scheduler.add_request(request)
    _step(scheduler, worker)
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    return request


def test_stuck_producer_is_fenced_and_every_blocked_request_freed(
    fenced_consumer, clock
):
    scheduler, connector, worker = fenced_consumer
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    stuck = _waiting_request(scheduler, worker, "xfer-stuck")
    clock.advance(TIMEOUT_S / 2)
    collateral = _waiting_request(scheduler, worker, "xfer-collateral")
    healthy = _waiting_request(scheduler, worker, "xfer-healthy", host="127.0.0.2")

    clock.advance(TIMEOUT_S / 2 + 1)
    results = _step(scheduler, worker)

    assert results.failed_recving == {stuck.request_id, collateral.request_id}
    assert stuck.status == RequestStatus.FINISHED_ERROR
    assert collateral.status == RequestStatus.FINISHED_ERROR
    assert stuck.request_id not in scheduler.requests
    assert collateral.request_id not in scheduler.requests
    # The other producer's request is untouched and completes normally.
    assert healthy.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    _step(scheduler, worker, finished_recving={healthy.request_id})
    _step(scheduler, worker)
    assert healthy.status == RequestStatus.RUNNING
    scheduler.finish_requests(healthy.request_id, RequestStatus.FINISHED_ABORTED)
    _step(scheduler, worker)
    assert pool.get_num_free_blocks() == free_before

    # Nothing is reported twice, and no timer fires for the fenced requests.
    clock.advance(10 * TIMEOUT_S)
    assert _step(scheduler, worker).finished_recving == set()
    assert connector._write_recv_deadline == {}


def test_unsafe_fence_fails_the_requests_but_keeps_their_blocks(fenced_consumer, clock):
    scheduler, connector, worker = fenced_consumer
    worker.moriio_engine.results = {
        moriio_engine_key("producer", "127.0.0.1", 4789, tp, 0): _fence_result(ok=False)
        for tp in range(2)
    }
    pool = scheduler.kv_cache_manager.block_pool
    stuck = _waiting_request(scheduler, worker, "xfer-stuck")
    collateral = _waiting_request(scheduler, worker, "xfer-collateral")
    held = pool.get_num_free_blocks()

    clock.advance(TIMEOUT_S + 1)
    _step(scheduler, worker)

    assert stuck.status == RequestStatus.FINISHED_ERROR
    assert stuck.request_id in scheduler.requests
    assert pool.get_num_free_blocks() == held
    # The collateral request was not failed by the fence; its own timer did.
    assert collateral.status == RequestStatus.FINISHED_ERROR
    assert collateral.request_id in scheduler.requests
