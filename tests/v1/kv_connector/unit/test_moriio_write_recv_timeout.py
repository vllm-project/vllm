# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO WRITE-mode consumer receive timeout (``write_recv_timeout``).

Runs the real ``Scheduler`` and the real ``MoRIIOConnectorWorker``
``start_load_kv`` / ``get_transfer_results``, with no GPU or RDMA.
"""

import time
from collections import OrderedDict
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

from .test_moriio_connector import _setup_kv_transfer_request
from .test_moriio_connector import create_vllm_config as create_moriio_config
from .utils import create_model_runner_output, create_request, create_scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

TIMEOUT_S = 60.0


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
    worker.moriio_wrapper = SimpleNamespace(pop_failed_write_req_ids=list)
    worker._unmatched_write_failures = OrderedDict()
    worker._write_recv_timed_out = set()
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
        self.pool = self.scheduler.kv_cache_manager.block_pool
        self.workers = [_make_consumer_worker() for _ in range(tp_size)]
        # Like the executors: only multi-worker outputs are aggregated.
        self.aggregator = KVOutputAggregator(tp_size) if tp_size > 1 else None

    def step(self, write_done: frozenset[str] = frozenset()):
        out = self.scheduler.schedule()
        scheduled = [self.scheduler.requests[i] for i in out.num_scheduled_tokens]
        outputs = []
        for worker in self.workers:
            worker.start_load_kv(out.kv_connector_metadata)
            worker.write_done |= write_done
            results = worker.get_transfer_results()
            output = create_model_runner_output(
                scheduled, finished_recving=results.finished_recving
            )
            output.kv_connector_output.failed_recving = results.failed_recving
            outputs.append(output)
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


@pytest.fixture(params=[1, 2], ids=["tp1", "tp2"])
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
    # write_done never comes, so the blocks are never released.
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

    # The late write_done for a finished request reaches the worker with #59058.
    engine.step(write_done=frozenset({request.request_id}))
    assert request.request_id not in engine.scheduler.requests
    assert engine.pool.get_num_free_blocks() == free_before
