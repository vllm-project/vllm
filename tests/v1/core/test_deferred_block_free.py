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
from unittest.mock import PropertyMock, patch

import pytest

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
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
    scheduler.defer_block_free = True
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
    # Overlapping batches + consumer-side connector enables the gate. Async
    # scheduling (which would give >1 concurrent batches) is force-disabled on
    # CPU, where this test runs, and PP can't be built without GPUs, so force
    # max_concurrent_batches to exercise the enabled path on any platform.
    with patch.object(
        VllmConfig,
        "max_concurrent_batches",
        new_callable=PropertyMock,
        return_value=2,
    ):
        scheduler = create_scheduler(
            model=MODEL,
            async_scheduling=True,
            use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
        )
    assert scheduler.defer_block_free


def test_gate_disabled_without_connector():
    # Async scheduling alone (no PD connector): the gate must stay off
    # and freeing must remain immediate.
    scheduler = create_scheduler(model=MODEL, async_scheduling=True)
    assert not scheduler.defer_block_free

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
    assert not scheduler.deferred_frees
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
    assert len(scheduler.deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running

    # Step 2's output is processed: every GPU write of step 2 has
    # completed, so the blocks can now be returned to the pool.
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert not scheduler.deferred_frees
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
    assert not scheduler.deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_abort_defers_free():
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request, out0, out1 = _setup_request_with_inflight_step(scheduler)
    num_free_running = pool.get_num_free_blocks()

    # External abort arrives while steps 1 and 2 are both in flight.
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert len(scheduler.deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running

    # Step 1's output: step 2 is still in flight, keep holding the blocks.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert len(scheduler.deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running

    # Step 2's output: now the blocks can be freed.
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert not scheduler.deferred_frees
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
    assert len(scheduler.deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        assert request.request_id not in manager.req_to_blocks

    # Outputs of both in-flight steps are processed: blocks return to the
    # pool only after the newest one.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert len(scheduler.deferred_frees) == 1
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert not scheduler.deferred_frees
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
    assert len(scheduler.deferred_frees) == 2
    assert pool.get_num_free_blocks() < num_free_initially

    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert not scheduler.deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_fence_held_across_multiple_inflight_steps():
    """Pipeline-parallel / deep async: with several steps scheduled ahead,
    a freed request's blocks must stay held until the *newest* in-flight
    step's output is processed, not the first.

    Depth-1 tests only check a single intervening update; with PP the
    scheduler can dispatch up to pp_size steps ahead, so the fence must
    survive multiple intervening update_from_output calls.
    """
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=10,
    )[0]
    scheduler.add_request(request)

    # Schedule three steps ahead without processing any output: a prefill
    # plus two speculatively over-scheduled decodes, all in flight at once.
    outs = [scheduler.schedule() for _ in range(3)]
    assert outs[0].num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS
    assert outs[1].num_scheduled_tokens[request.request_id] == 1
    assert outs[2].num_scheduled_tokens[request.request_id] == 1
    assert scheduler.sched_step_seq == 3
    num_free_running = pool.get_num_free_blocks()
    assert num_free_running < num_free_initially

    # Abort while all three steps are in flight: the fence is the newest
    # scheduled step (3), since any of them may still write the blocks.
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert len(scheduler.deferred_frees) == 1
    assert scheduler.deferred_frees[0][0] == 3
    assert pool.get_num_free_blocks() == num_free_running

    # Draining the two earlier in-flight steps must NOT release the blocks:
    # their outputs don't fence the still-pending newest write.
    for out in (outs[0], outs[1]):
        scheduler.update_from_output(out, _make_model_runner_output(out))
        assert len(scheduler.deferred_frees) == 1
        assert pool.get_num_free_blocks() == num_free_running

    # Only once the newest scheduled step's output is processed do the
    # blocks return to the pool.
    scheduler.update_from_output(outs[2], _make_model_runner_output(outs[2]))
    assert not scheduler.deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_max_tokens_finish_frees_immediately_with_other_inflight():
    """A request finishing by reaching max_tokens is never over-scheduled past
    its final-token step, so no in-flight step writes its blocks: it is freed
    immediately even while another request's step is still in flight.
    """
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool

    # Short request finishes at max_tokens=1; long request keeps running.
    short = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, max_tokens=1, req_ids=["short"]
    )[0]
    long = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, max_tokens=100, req_ids=["long"]
    )[0]
    scheduler.add_request(short)
    scheduler.add_request(long)

    out0 = scheduler.schedule()  # prefill both
    out1 = scheduler.schedule()  # short is skipped (at max_tokens); long decodes
    assert "short" not in out1.num_scheduled_tokens
    assert "long" in out1.num_scheduled_tokens

    free_before = pool.get_num_free_blocks()
    # Process step 0: `short` reaches max_tokens and finishes while step 1
    # (which scheduled `long`, not `short`) is still in flight.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))

    assert short.is_finished()
    # A step IS globally in flight (the old global fence would have deferred),
    # but the per-request gate frees `short` immediately since nothing writes
    # its blocks anymore.
    assert scheduler.sched_step_seq > scheduler.processed_step_seq
    assert not scheduler.deferred_frees
    assert pool.get_num_free_blocks() > free_before  # short's blocks returned


def test_abort_mid_prefill_defers_free():
    """Intermediate prefill chunks don't allocate output placeholders, so the
    deferral must key off is_prefill_chunk: aborting a request whose prefill
    chunk is still in flight must withhold its blocks.
    """
    scheduler = create_scheduler(
        model=MODEL, async_scheduling=True, long_prefill_token_threshold=16
    )
    scheduler.defer_block_free = True
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, max_tokens=5
    )[0]
    scheduler.add_request(request)

    out0 = scheduler.schedule()
    # Partial prefill: a chunk is in flight, with no output placeholders yet.
    assert out0.num_scheduled_tokens[request.request_id] == 16
    assert request.num_output_placeholders == 0
    assert request.is_prefill_chunk
    num_free_running = pool.get_num_free_blocks()
    assert num_free_running < num_free_initially

    # Abort while the prefill chunk is in flight: blocks must be withheld
    # (keyed off is_prefill_chunk, since there are no placeholders).
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert len(scheduler.deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running

    # Once the in-flight prefill step's output is processed, blocks return.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert not scheduler.deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_non_async_abort_defers_via_last_sched_seq():
    """Without async (e.g. PP filling the pipeline) there are no placeholders
    and a full prefill isn't a partial chunk, yet an abort with a step in flight
    must defer. Only the last-scheduled-step fence catches this.

    PP=2 can't be built on a single-GPU host, so force the flag and exercise the
    mechanism; the gate itself is covered by test_gate_enabled_for_async_consumer.
    """
    scheduler = create_scheduler(model=MODEL, async_scheduling=False)
    scheduler.defer_block_free = True
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, max_tokens=5
    )[0]
    scheduler.add_request(request)

    out0 = scheduler.schedule()
    # Neither async-only signal marks this request as in flight.
    assert request.num_output_placeholders == 0
    assert not request.is_prefill_chunk
    # Only the last-scheduled-step fence does.
    assert request.last_sched_seq > scheduler.processed_step_seq
    num_free_running = pool.get_num_free_blocks()
    assert num_free_running < num_free_initially

    # Abort while out0 is in flight: blocks must be withheld.
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert len(scheduler.deferred_frees) == 1
    assert pool.get_num_free_blocks() == num_free_running

    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert not scheduler.deferred_frees
    assert pool.get_num_free_blocks() == num_free_initially


def test_cow_retentions_deferred_until_copy_step_processed():
    """The endpoints of a queued KV block copy must stay out of the free
    pool until the step that runs the copy has been processed. Freed
    earlier, an endpoint can be reallocated (e.g. as a PD KV-load
    destination) and overwritten by a transfer that is not ordered against
    the copy still pending in the in-flight step.
    """
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]

    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=5,
        stop_token_ids=[STOP_TOKEN_ID],
    )[0]
    scheduler.add_request(request)

    # Simulate a partial-hit CoW performed while scheduling step 1, whose
    # hitting request was freed within the same step: the copy rides out0
    # and each endpoint stays alive only through its copy retention.
    src_block, dst_block = pool.get_new_blocks(2)
    block_copy = KVCacheBlockCopy(
        src_block_id=src_block.block_id, dst_block_id=dst_block.block_id
    )
    manager._kv_cache_block_copies.append(block_copy)
    out0 = scheduler.schedule()
    assert out0.kv_cache_block_copies == [block_copy]
    manager._cow_blocks_to_release.extend((src_block, dst_block))

    # Exhaust the rest of the pool so the copy endpoints are the only blocks
    # a new request could receive, then add one that fits exactly in them.
    pool.get_new_blocks(pool.get_num_free_blocks())
    late_request = create_requests(
        num_requests=1,
        num_tokens=2 * scheduler.block_size,
        max_tokens=5,
        req_ids=["late"],
    )[0]
    scheduler.add_request(late_request)

    # Step 2 is scheduled while step 1 (which runs the copy) is still in
    # flight: the retentions are released against the copy's fence, so the
    # endpoints must not reach the free pool -- the late request must not be
    # scheduled onto them.
    out1 = scheduler.schedule()
    assert src_block.ref_cnt == 1
    assert dst_block.ref_cnt == 1
    assert scheduler.deferred_frees
    assert not out1.scheduled_new_reqs

    # Step 1's output is processed: the copy has run, endpoints return to
    # the pool and the late request can be scheduled onto them safely.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert src_block.ref_cnt == 0
    assert dst_block.ref_cnt == 0
    assert not scheduler.deferred_frees
    out2 = scheduler.schedule()
    assert [r.req_id for r in out2.scheduled_new_reqs] == ["late"]
