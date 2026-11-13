# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO WRITE-mode consumer receive timeout (``write_recv_timeout``).

Runs the real ``Scheduler`` and the real ``MoRIIOConnectorWorker``
``start_load_kv`` / ``get_transfer_results``, with no GPU or RDMA.
"""

import socket
import time
from collections import OrderedDict
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_common,
    moriio_connector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.v1.request import RequestStatus

from ..utils import create_model_runner_output, create_request, create_scheduler
from .test_moriio_connector import _setup_kv_transfer_request
from .test_moriio_connector import create_vllm_config as create_moriio_config

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

TIMEOUT_S = 60.0


class _InlineExecutor:
    def submit(self, fn, *args):
        future: Future = Future()
        future.set_result(fn(*args))
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


class _Clock:
    def __init__(self):
        self.now = time.monotonic()

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


def _make_consumer_worker() -> MoRIIOConnectorWorker:
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.WRITE
    worker.write_failed = set()  # transfer ids of producer write_failed messages

    def pop_failed_write_req_ids():
        failed, worker.write_failed = worker.write_failed, set()
        return list(failed)

    worker.moriio_wrapper = SimpleNamespace(
        pop_failed_write_req_ids=pop_failed_write_req_ids, shutdown=lambda: None
    )
    worker._unmatched_write_failures = OrderedDict()
    worker._write_recv_timed_out = set()
    worker._write_recv_reported = set()
    worker._write_recv_unreleased = set()
    worker.write_done = set()

    def get_finished():
        done, worker.write_done = worker.write_done, set()
        return set(), done

    worker.get_finished = get_finished  # type: ignore[method-assign]
    return worker


class _Engine:
    """A decode instance: the real scheduler and one stub worker per TP rank."""

    def __init__(self, monkeypatch, tp_size: int):
        self.clock = _Clock()
        monkeypatch.setattr(moriio_connector, "time", self.clock)
        config = create_moriio_config(role="kv_consumer")
        config.kv_transfer_config.kv_connector_extra_config["write_recv_timeout"] = (
            TIMEOUT_S
        )
        self.scheduler = create_scheduler(config)
        connector = self.scheduler.connector.connector_scheduler
        monkeypatch.setattr(connector, "send_notify_block", lambda **kwargs: None)
        # The prefill's notify socket: gone unless a test says it answers. Probes
        # answer at once, so a deadline is decided in the step it passes.
        self.prefill_alive = False
        monkeypatch.setattr(
            moriio_connector, "_peer_listening", lambda host, port: self.prefill_alive
        )
        connector._write_recv_prober.shutdown()
        connector._write_recv_prober = _InlineExecutor()
        self.pool = self.scheduler.kv_cache_manager.block_pool
        self.workers = [_make_consumer_worker() for _ in range(tp_size)]
        # Like the executors: only multi-worker outputs are aggregated.
        self.aggregator = (
            KVOutputAggregator.from_connector(self.scheduler.connector, tp_size)
            if tp_size > 1
            else None
        )

    def step(self, write_done=frozenset(), ranks=None, write_failed=frozenset()):
        self.finish(self.scheduler.schedule(), write_done, ranks, write_failed)

    def finish(self, out, write_done=frozenset(), ranks=None, write_failed=frozenset()):
        """Runs one step's workers. `write_done` (request ids) and `write_failed`
        (transfer ids) reach the TP `ranks` (all)."""
        scheduled = [self.scheduler.requests[i] for i in out.num_scheduled_tokens]
        outputs = []
        for rank, worker in enumerate(self.workers):
            worker.start_load_kv(out.kv_connector_metadata)
            if ranks is None or rank in ranks:
                worker.write_done |= write_done
                worker.write_failed |= write_failed
            results = worker.get_transfer_results()
            outputs.append(
                create_model_runner_output(
                    scheduled,
                    finished_sending=results.finished_sending,
                    finished_recving=results.finished_recving,
                    failed_recving=results.failed_recving,
                )
            )
        if self.aggregator is not None:
            outputs = [self.aggregator.aggregate(outputs)]
        self.scheduler.update_from_output(out, outputs[0])

    def add_waiting_request(self):
        request = create_request(
            num_tokens=4 * self.scheduler.block_size, do_remote_prefill=True
        )
        _setup_kv_transfer_request(request)
        self.scheduler.add_request(request)
        self.step()
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        return request


@pytest.fixture(params=[1, 2, 4], ids=["tp1", "tp2", "tp4"])
def engine(request, monkeypatch):
    role = moriio_common.get_role()
    moriio_common.set_role(moriio_common.ROLE.CONSUMER)
    yield _Engine(monkeypatch, request.param)
    moriio_common.set_role(role)


def test_prefill_dies_request_fails_and_blocks_stay_allocated(engine):
    free_before = engine.pool.get_num_free_blocks()
    request = engine.add_waiting_request()
    held = engine.pool.get_num_free_blocks()
    assert held < free_before

    engine.clock.now += TIMEOUT_S - 1
    engine.step()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    engine.clock.now += 2
    engine.step()
    assert request.status == RequestStatus.FINISHED_ERROR
    engine.clock.now += 10 * TIMEOUT_S
    for _ in range(3):
        engine.step()
    assert request.request_id in engine.scheduler.requests
    assert engine.pool.get_num_free_blocks() == held


def test_prefill_slow_request_fails_and_blocks_freed_on_late_write_done(engine):
    free_before = engine.pool.get_num_free_blocks()
    request = engine.add_waiting_request()
    held = engine.pool.get_num_free_blocks()

    engine.clock.now += TIMEOUT_S + 1
    engine.step()
    assert request.status == RequestStatus.FINISHED_ERROR
    assert engine.pool.get_num_free_blocks() == held

    engine.step(write_done=frozenset({request.request_id}))
    assert request.request_id not in engine.scheduler.requests
    assert engine.pool.get_num_free_blocks() == free_before


def test_timeout_reported_once_when_two_schedules_precede_the_output(engine):
    """Batch queue / async scheduling: step N+1 is scheduled before the output
    of step N is processed."""
    request = engine.add_waiting_request()
    held = engine.pool.get_num_free_blocks()
    engine.clock.now += TIMEOUT_S + 1

    first = engine.scheduler.schedule()
    second = engine.scheduler.schedule()
    assert first.kv_connector_metadata.write_recv_timed_out_req_ids == {
        request.request_id
    }
    assert not second.kv_connector_metadata.write_recv_timed_out_req_ids

    engine.finish(first)
    engine.finish(second)
    assert request.status == RequestStatus.FINISHED_ERROR
    assert engine.pool.get_num_free_blocks() == held


def test_blocks_freed_once_every_rank_has_its_write_done(engine):
    if len(engine.workers) < 2:
        pytest.skip("needs more than one TP rank")
    free_before = engine.pool.get_num_free_blocks()
    request = engine.add_waiting_request()
    held = engine.pool.get_num_free_blocks()

    # The even ranks' writes land before the timeout, the odd ranks' after it.
    ranks = range(len(engine.workers))
    engine.step(write_done=frozenset({request.request_id}), ranks=set(ranks[::2]))
    engine.clock.now += TIMEOUT_S + 1
    engine.step()
    assert request.status == RequestStatus.FINISHED_ERROR
    assert engine.pool.get_num_free_blocks() == held

    engine.step(write_done=frozenset({request.request_id}), ranks=set(ranks[1::2]))
    assert request.request_id not in engine.scheduler.requests
    assert engine.pool.get_num_free_blocks() == free_before


def test_write_done_with_the_timeout_completes_the_request(engine):
    request = engine.add_waiting_request()
    engine.clock.now += TIMEOUT_S + 1

    engine.step(write_done=frozenset({request.request_id}))
    engine.step()
    assert request.status == RequestStatus.RUNNING


def test_alive_prefill_is_waited_for_past_the_deadline(engine):
    """A prefill that still answers at the deadline is slow, e.g. behind a long
    queue, not gone: its request keeps waiting and completes."""
    engine.prefill_alive = True
    request = engine.add_waiting_request()

    engine.clock.now += TIMEOUT_S + 1
    engine.step()
    engine.clock.now += TIMEOUT_S
    engine.step()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    engine.step(write_done=frozenset({request.request_id}))
    engine.step()
    assert request.status == RequestStatus.RUNNING


def test_alive_prefill_still_fails_at_the_max_timeout(engine):
    engine.prefill_alive = True
    request = engine.add_waiting_request()
    held = engine.pool.get_num_free_blocks()

    engine.clock.now += TIMEOUT_S + 1
    engine.step()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # write_recv_max_timeout defaults to 3 x write_recv_timeout.
    engine.clock.now += 2 * TIMEOUT_S
    engine.step()
    assert request.status == RequestStatus.FINISHED_ERROR
    assert engine.pool.get_num_free_blocks() == held


def test_peer_listening_tells_a_listening_socket_from_a_closed_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert moriio_connector._peer_listening("127.0.0.1", port)
    assert not moriio_connector._peer_listening("127.0.0.1", port)


def test_rank_that_reported_write_failed_releases_at_the_timeout(engine):
    """A rank whose producer reported write_failed has reported the receive; at
    the timeout it releases it instead of reporting it a second time."""
    if len(engine.workers) < 2:
        pytest.skip("needs more than one TP rank")
    free_before = engine.pool.get_num_free_blocks()
    request = engine.add_waiting_request()
    held = engine.pool.get_num_free_blocks()
    transfer_id = request.kv_transfer_params["transfer_id"]

    engine.step(write_failed=frozenset({transfer_id}), ranks={0})
    engine.clock.now += TIMEOUT_S + 1
    engine.step()
    assert request.status == RequestStatus.FINISHED_ERROR
    assert engine.pool.get_num_free_blocks() == held

    late_ranks = set(range(1, len(engine.workers)))
    engine.step(write_done=frozenset({request.request_id}), ranks=late_ranks)
    assert request.request_id not in engine.scheduler.requests
    assert engine.pool.get_num_free_blocks() == free_before


def test_request_aborted_before_its_deadline_stops_its_timer(engine):
    request = engine.add_waiting_request()
    engine.scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    engine.clock.now += TIMEOUT_S + 1
    out = engine.scheduler.schedule()
    assert not out.kv_connector_metadata.write_recv_timed_out_req_ids
    engine.finish(out)


def test_peer_listening_treats_a_malformed_host_as_not_listening():
    # IDNA rejects a label over 63 characters with UnicodeError, not OSError.
    assert not moriio_connector._peer_listening("a" * 64 + ".invalid", 1)
