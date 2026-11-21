# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO WRITE-mode consumer receive timeout (`write_recv_timeout`).

Runs the real `Scheduler` and the real `MoRIIOConnectorWorker`
`start_load_kv` / `get_transfer_results`, with no GPU or RDMA, at TP 2 through
the real `KVOutputAggregator`; the reproducer runs one worker without it. One
test covers the hybrid decode KV cache layouts: attention or MLA, each with a
Mamba/KDA state group.
"""

import math
import time
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

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
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import RequestStatus

from ..utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    make_kv_cache_group,
)
from .test_moriio_connector import _setup_kv_transfer_request
from .test_moriio_connector import create_vllm_config as create_moriio_config

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

TIMEOUT_S = 60.0
TP_SIZE = 2
BLOCK_SIZE = 16
# Attention or MLA, each with a Mamba/KDA state group.
LAYOUTS = ["hybrid", "mla_hybrid"]


def _kv_cache_config(layout: str) -> KVCacheConfig:
    """The decode's KV cache groups: one attention group, MLA for the `mla`
    layouts, plus a Mamba/KDA state group for the `hybrid` ones."""
    if layout.startswith("mla"):
        attn = MLAAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
        )
    else:
        attn = AttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=4, head_size=16, dtype=torch.bfloat16
        )
    groups = [make_kv_cache_group(["layers.0.attn"], attn)]
    if layout.endswith("hybrid"):
        state = MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((3, 16), (4, 16, 16)),
            dtypes=(torch.bfloat16, torch.float32),
        )
        groups.append(
            make_kv_cache_group(
                ["layers.1.linear_attn"],
                state,
                MambaManager,
                MambaManager.config(enable_prefix_caching=True),
            )
        )
    return KVCacheConfig(num_blocks=10000, kv_cache_tensors=[], kv_cache_groups=groups)


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
    worker.write_done = set()

    def get_finished():
        done, worker.write_done = worker.write_done, set()
        return set(), done

    worker.get_finished = get_finished  # type: ignore[method-assign]
    return worker


class _Engine:
    """A decode instance: the real scheduler and one stub worker per TP rank."""

    def __init__(self, monkeypatch, layout: str = "attn"):
        self.clock = _Clock()
        monkeypatch.setattr(moriio_connector, "time", self.clock)
        config = create_moriio_config(role="kv_consumer", block_size=BLOCK_SIZE)
        config.kv_transfer_config.kv_connector_extra_config["write_recv_timeout"] = (
            TIMEOUT_S
        )
        kv_cache_config = _kv_cache_config(layout)
        self.scheduler = create_scheduler(config, kv_cache_config=kv_cache_config)
        self.num_groups = len(kv_cache_config.kv_cache_groups)
        connector = self.scheduler.connector.connector_scheduler
        monkeypatch.setattr(connector, "send_notify_block", lambda **kwargs: None)
        self.pool = self.scheduler.kv_cache_manager.block_pool
        self.workers = [_make_consumer_worker() for _ in range(TP_SIZE)]
        self.aggregator = KVOutputAggregator.from_connector(
            self.scheduler.connector, TP_SIZE
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
        self.scheduler.update_from_output(out, self.aggregator.aggregate(outputs))

    def add_waiting_request(self):
        request = create_request(
            num_tokens=4 * self.scheduler.block_size, do_remote_prefill=True
        )
        _setup_kv_transfer_request(request)
        self.scheduler.add_request(request)
        self.step()
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        # Every KV cache group holds blocks for the receive.
        block_ids = self.scheduler.kv_cache_manager.get_block_ids(request.request_id)
        assert len(block_ids) == self.num_groups and all(block_ids)
        return request


@pytest.fixture
def consumer_role():
    role = moriio_common.get_role()
    moriio_common.set_role(moriio_common.ROLE.CONSUMER)
    yield
    moriio_common.set_role(role)


@pytest.fixture
def engine(consumer_role, monkeypatch):
    return _Engine(monkeypatch)


def test_dead_prefill_does_not_leave_its_request_waiting(consumer_role, monkeypatch):
    """#59114: the prefill dies after the decode allocated the request's blocks,
    so no `write_done` ever arrives. Without the timer the request waits in
    `WAITING_FOR_REMOTE_KVS` forever. Uses nothing that exists only with the
    timer, so it also runs, and fails, without it."""
    clock = _Clock()
    monkeypatch.setattr(moriio_connector, "time", clock)
    config = create_moriio_config(role="kv_consumer", block_size=BLOCK_SIZE)
    config.kv_transfer_config.kv_connector_extra_config["write_recv_timeout"] = (
        TIMEOUT_S
    )
    scheduler = create_scheduler(config)
    monkeypatch.setattr(
        scheduler.connector.connector_scheduler,
        "send_notify_block",
        lambda **kwargs: None,
    )
    worker = _make_consumer_worker()

    def step():
        out = scheduler.schedule()
        worker.start_load_kv(out.kv_connector_metadata)
        results = worker.get_transfer_results()
        scheduler.update_from_output(
            out,
            create_model_runner_output(
                [scheduler.requests[i] for i in out.num_scheduled_tokens],
                finished_sending=results.finished_sending,
                finished_recving=results.finished_recving,
                failed_recving=results.failed_recving,
            ),
        )

    request = create_request(num_tokens=4 * BLOCK_SIZE, do_remote_prefill=True)
    _setup_kv_transfer_request(request)
    scheduler.add_request(request)
    step()
    pool = scheduler.kv_cache_manager.block_pool
    held = pool.get_num_free_blocks()
    clock.now += TIMEOUT_S - 1
    step()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    clock.now += 60 * TIMEOUT_S
    for _ in range(3):
        step()
    assert request.status == RequestStatus.FINISHED_ERROR
    # A late write could still land in its blocks, so they stay allocated.
    assert request.request_id in scheduler.requests
    assert pool.get_num_free_blocks() == held


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


def test_timer_is_dropped_when_its_receive_finishes(engine):
    """The timer lives on its receive's record, so it goes when the receive
    finishes, not when the process exits."""
    connector = engine.scheduler.connector.connector_scheduler
    request = engine.add_waiting_request()
    assert math.isfinite(connector._write_recvs_in_flight[request.request_id].deadline)

    engine.step(write_done=frozenset({request.request_id}))
    engine.step()
    assert request.status == RequestStatus.RUNNING
    assert request.request_id not in connector._write_recvs_in_flight


def test_request_aborted_before_its_deadline_stops_its_timer(engine):
    request = engine.add_waiting_request()
    engine.scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    engine.clock.now += TIMEOUT_S + 1
    out = engine.scheduler.schedule()
    assert not out.kv_connector_metadata.write_recv_timed_out_req_ids
    engine.finish(out)


def test_write_failed_with_the_timeout_frees_the_blocks(engine):
    """A producer write_failed that reaches the workers in the step the timeout
    does: the write is over, so the blocks must be freed, not held forever."""
    free_before = engine.pool.get_num_free_blocks()
    request = engine.add_waiting_request()
    transfer_id = request.kv_transfer_params["transfer_id"]

    engine.clock.now += TIMEOUT_S + 1
    engine.step(write_failed=frozenset({transfer_id}))
    assert request.status == RequestStatus.FINISHED_ERROR
    engine.step()
    assert request.request_id not in engine.scheduler.requests
    assert engine.pool.get_num_free_blocks() == free_before


@pytest.mark.parametrize("role, timed", [("kv_consumer", True), ("kv_both", False)])
def test_timer_runs_only_on_a_pure_consumer(consumer_role, role, timed):
    """A kv_both scheduler acts as a consumer, but its workers run as producers
    and never report a timed-out receive, so its timer stays off."""
    config = create_moriio_config(role=role, block_size=BLOCK_SIZE)
    connector = create_scheduler(config).connector.connector_scheduler
    assert (connector._write_recv_timeout is not None) == timed


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_kv_cache_group_is_held_and_then_freed(
    consumer_role, monkeypatch, layout
):
    """The failure is per request and the hold covers all of its blocks, in
    every KV cache group (attention or MLA, with or without Mamba/KDA state)."""
    engine = _Engine(monkeypatch, layout)
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
