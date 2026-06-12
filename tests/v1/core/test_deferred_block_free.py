# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for deferred block freeing under async scheduling.

With async scheduling, a finished/preempted request's blocks may still be
written by a speculatively over-scheduled in-flight GPU step (mamba/GDN
layers rewrite the whole state block every step). If such a block is
reallocated to a request arriving via PD disaggregation, the NIC/RDMA write
of the received state races with the in-flight stale write. The scheduler
closes the race by deferring the return of blocks to the block pool until
the newest scheduled step's output has been processed.
"""

import os
import time

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test

# Allow overriding the model with a local path for offline environments.
MODEL = os.environ.get("VLLM_TEST_DEFER_FREE_MODEL", "facebook/opt-125m")
STOP_TOKEN_ID = 42
NUM_PROMPT_TOKENS = 33  # 3 blocks with block_size=16


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
    token_id: int = 0,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _create_deferring_scheduler():
    """Async scheduler with deferred block freeing forced on.

    The production gate additionally requires a PD KV-consumer connector;
    the mechanism itself is independent of it.
    """
    scheduler = create_scheduler(model=MODEL, async_scheduling=True)
    scheduler._defer_block_free = True
    return scheduler


def _setup_request_with_inflight_step(scheduler, max_tokens: int = 5):
    """Schedule a request's prefill (step 1) and one speculatively
    over-scheduled decode (step 2), mimicking async scheduling depth 1.

    Returns (request, out0, out1).
    """
    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=max_tokens,
        stop_token_ids=[STOP_TOKEN_ID],
    )[0]
    scheduler.add_request(request)
    out0 = scheduler.schedule()
    assert out0.num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens[request.request_id] == 1
    return request, out0, out1


def test_gate_enabled_for_async_consumer():
    # Async scheduling + consumer-side connector: the gate is on even
    # without mamba layers (the in-flight-write vs NIC-write race also
    # corrupts full-attention KV, just with a smaller blast radius).
    scheduler = create_scheduler(
        model=MODEL,
        async_scheduling=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    assert scheduler._defer_block_free


def test_gate_disabled_without_connector():
    # Async scheduling alone (no PD connector): the gate must stay off
    # and freeing must remain immediate.
    scheduler = create_scheduler(model=MODEL, async_scheduling=True)
    assert not scheduler._defer_block_free

    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request, out0, out1 = _setup_request_with_inflight_step(scheduler)
    assert pool.get_num_free_blocks() < num_free_initially

    # Request stops early while step 2 is in flight: blocks are freed
    # immediately because deferral is disabled.
    scheduler.update_from_output(
        out0, _make_model_runner_output(out0, token_id=STOP_TOKEN_ID)
    )
    assert request.is_finished()
    assert not scheduler._deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_finish_defers_free_until_inflight_step_done():
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request, out0, out1 = _setup_request_with_inflight_step(scheduler)
    num_free_running = pool.get_num_free_blocks()
    assert num_free_running < num_free_initially

    # The request stops early (stop token) while the over-scheduled step 2
    # is still in flight: its blocks must NOT return to the pool yet.
    scheduler.update_from_output(
        out0, _make_model_runner_output(out0, token_id=STOP_TOKEN_ID)
    )
    assert request.is_finished()
    assert len(scheduler._deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running

    # Step 2's output is processed: every GPU write of step 2 has
    # completed, so the blocks can now be returned to the pool.
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert not scheduler._deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_finish_frees_immediately_when_no_inflight_step():
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=5,
        stop_token_ids=[STOP_TOKEN_ID],
    )[0]
    scheduler.add_request(request)
    out0 = scheduler.schedule()

    # Synchronous-like flow: out0 is the newest scheduled step and its
    # output is being processed, so no other step can still write the
    # blocks and the free happens immediately.
    scheduler.update_from_output(
        out0, _make_model_runner_output(out0, token_id=STOP_TOKEN_ID)
    )
    assert request.is_finished()
    assert not scheduler._deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_abort_defers_free():
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request, out0, out1 = _setup_request_with_inflight_step(scheduler)
    num_free_running = pool.get_num_free_blocks()

    # External abort arrives while steps 1 and 2 are both in flight.
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert len(scheduler._deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running

    # Step 1's output: step 2 is still in flight, keep holding the blocks.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert len(scheduler._deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running

    # Step 2's output: now the blocks can be freed.
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert not scheduler._deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_preempt_defers_free_and_clears_bookkeeping():
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request, out0, out1 = _setup_request_with_inflight_step(scheduler)
    num_free_running = pool.get_num_free_blocks()

    # Preempt the request while steps are in flight (mirrors the
    # preemption path inside schedule()).
    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.monotonic())
    assert request.status == RequestStatus.PREEMPTED

    # Blocks are withheld from the pool, but the manager bookkeeping is
    # cleared immediately so the request can be rescheduled safely.
    assert len(scheduler._deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        assert request.request_id not in manager.req_to_blocks

    # Outputs of both in-flight steps are processed: blocks return to the
    # pool only after the newest one.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert len(scheduler._deferred_frees) == 1
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert not scheduler._deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_multiple_deferred_frees_drain_in_order():
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    requests = create_requests(
        num_requests=2,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=5,
        stop_token_ids=[STOP_TOKEN_ID],
    )
    for request in requests:
        scheduler.add_request(request)
    out0 = scheduler.schedule()
    out1 = scheduler.schedule()

    # Both requests stop early at step 1's output while step 2 is in
    # flight: two deferred entries with the same fence.
    scheduler.update_from_output(
        out0, _make_model_runner_output(out0, token_id=STOP_TOKEN_ID)
    )
    assert len(scheduler._deferred_frees) == 2
    assert pool.get_num_free_blocks() < num_free_initially

    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert not scheduler._deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially
