# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for drain-before-reuse (Percy step ledger, A2).

With overlapping batches (async scheduling, PP), a finished/preempted
request's blocks may still be read or written by an in-flight GPU step
(mamba/GDN layers rewrite the whole state block every step). If such a block
were reallocated -- to any request, or as a PD KV-load destination -- the new
owner's writes would race the in-flight stale ones. Every release is stamped
with the last step that may still use the block, and the GrantTable keeps it
DRAINING (out of the free queue) until that step's output has been
processed. This holds unconditionally, with or without a KV connector.
"""

import os
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import PrefixCacheStats
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
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _create_deferring_scheduler(scheduling_policy="fcfs"):
    """Async scheduler; drain-before-reuse needs no flag or connector."""
    return create_scheduler(
        model=MODEL,
        async_scheduling=True,
        scheduling_policy=scheduling_policy,
    )


def _num_draining(scheduler) -> int:
    """Blocks released but withheld from reuse until their step is processed."""
    return scheduler.kv_cache_manager.num_draining_blocks


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


def _fail_one_allocation(scheduler, call_number: int):
    real_allocate = scheduler.kv_cache_manager.allocate_slots
    calls = 0

    def allocate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == call_number:
            return None
        return real_allocate(*args, **kwargs)

    return patch.object(
        scheduler.kv_cache_manager,
        "allocate_slots",
        side_effect=allocate,
    )


def test_drains_with_async_consumer_connector():
    # Overlapping batches + consumer-side connector (the only case the old
    # gate covered) still withholds blocks. Async scheduling is force-disabled
    # on CPU, where this test runs, and PP can't be built without GPUs, so
    # force max_concurrent_batches.
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
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()
    request, out0, out1 = _setup_request_with_inflight_step(scheduler)
    num_free_running = pool.get_num_free_blocks()

    scheduler.update_from_output(
        out0, _make_model_runner_output(out0, token_id=STOP_TOKEN_ID)
    )
    assert request.is_finished()
    assert _num_draining(scheduler) == num_free_initially - num_free_running
    assert pool.get_num_free_blocks() == num_free_running
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert _num_draining(scheduler) == 0
    assert pool.get_num_free_blocks() == num_free_initially


def test_drains_without_connector():
    """Spec A2 acceptance: with batch queue depth 2 and no connector, a block
    freed in step N is not re-granted before step N is processed (the old gate
    freed immediately here, racing the in-flight step)."""
    # The null block plus exactly the 3 blocks the request holds.
    scheduler = create_scheduler(model=MODEL, async_scheduling=True, num_blocks=4)
    assert scheduler.connector is None
    pool = scheduler.kv_cache_manager.block_pool
    table = scheduler.kv_cache_manager.grant_table
    num_free_initially = pool.get_num_free_blocks()

    request, out0, out1 = _setup_request_with_inflight_step(scheduler)
    num_free_running = pool.get_num_free_blocks()
    freed_ids = [
        b.block_id
        for b in scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[0]
    ]

    # Request stops early while step 2 (which also scheduled it) is in flight.
    scheduler.update_from_output(
        out0, _make_model_runner_output(out0, token_id=STOP_TOKEN_ID)
    )
    assert request.is_finished()
    assert _num_draining(scheduler) == len(freed_ids)
    assert pool.get_num_free_blocks() == num_free_running

    # The in-flight blocks are the only ones left, and none may be granted:
    # a request needing one gets no room (not scheduled), not an exception.
    assert pool.get_num_free_blocks() == 0
    hungry = create_requests(
        num_requests=1,
        num_tokens=scheduler.block_size,
        max_tokens=1,
        req_ids=["hungry"],
    )[0]
    scheduler.add_request(hungry)
    out2 = scheduler.schedule()
    assert "hungry" not in out2.num_scheduled_tokens
    assert all(table.state(i).name == "DRAINING" for i in freed_ids)

    # Step 2 is processed: the blocks drain and become reusable.
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert _num_draining(scheduler) == 0
    assert all(table.state(i).name == "FREE" for i in freed_ids)
    assert pool.get_num_free_blocks() == num_free_initially
    assert "hungry" in scheduler.schedule().num_scheduled_tokens


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
    assert _num_draining(scheduler) > 0
    assert pool.get_num_free_blocks() == num_free_running

    # Step 2's output is processed: every GPU write of step 2 has
    # completed, so the blocks can now be returned to the pool.
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert _num_draining(scheduler) == 0
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
    assert _num_draining(scheduler) == 0
    assert pool.get_num_free_blocks() == num_free_initially


def test_abort_defers_free():
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()

    request, out0, out1 = _setup_request_with_inflight_step(scheduler)
    num_free_running = pool.get_num_free_blocks()

    # External abort arrives while steps 1 and 2 are both in flight.
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert _num_draining(scheduler) > 0
    assert pool.get_num_free_blocks() == num_free_running

    # Step 1's output: step 2 is still in flight, keep holding the blocks.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert _num_draining(scheduler) > 0
    assert pool.get_num_free_blocks() == num_free_running

    # Step 2's output: now the blocks can be freed.
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert _num_draining(scheduler) == 0
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
    assert _num_draining(scheduler) > 0
    assert pool.get_num_free_blocks() == num_free_running
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        assert request.request_id not in manager.req_to_blocks

    # Outputs of both in-flight steps are processed: blocks return to the
    # pool only after the newest one.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert _num_draining(scheduler) > 0
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert _num_draining(scheduler) == 0
    assert pool.get_num_free_blocks() == num_free_initially


@pytest.mark.parametrize(
    ("policy", "failed_call"),
    [
        ("fcfs", 1),
        ("priority", 2),
    ],
)
def test_allocation_retry_waits_for_fence_then_succeeds(policy, failed_call):
    scheduler = _create_deferring_scheduler(policy)
    requests = create_requests(
        num_requests=3,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=10,
        stop_token_ids=[STOP_TOKEN_ID],
    )
    for request in requests:
        scheduler.add_request(request)
    out0 = scheduler.schedule()
    out1 = scheduler.schedule()

    worst, trigger, tail = requests
    worst.priority, trigger.priority, tail.priority = 9, 0, 1

    with _fail_one_allocation(scheduler, failed_call) as allocate:
        blocked = scheduler.schedule()
    assert allocate.call_count == failed_call
    assert not blocked.preempted_req_ids
    retry_trigger = requests[failed_call - 1]
    assert retry_trigger.request_id not in blocked.num_scheduled_tokens

    for output in (out0, out1, blocked):
        if output.total_num_scheduled_tokens:
            scheduler.update_from_output(
                output,
                _make_model_runner_output(output),
            )

    expected_victim = worst if policy == "priority" else tail
    with _fail_one_allocation(scheduler, failed_call) as allocate:
        resumed = scheduler.schedule()

    assert allocate.call_count > failed_call
    assert expected_victim.request_id in resumed.preempted_req_ids
    assert retry_trigger.request_id in resumed.num_scheduled_tokens
    if policy == "priority":
        assert tail.request_id in resumed.num_scheduled_tokens


def test_multiple_released_requests_drain_together():
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
    assert _num_draining(scheduler) > 0
    assert pool.get_num_free_blocks() < num_free_initially

    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert _num_draining(scheduler) == 0
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

    # Abort while all three steps are in flight: the stamp is the newest
    # step that scheduled the request (3), since any of them may still write
    # the blocks.
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert _num_draining(scheduler) > 0
    assert request.last_sched_seq == 3
    assert pool.get_num_free_blocks() == num_free_running

    # Draining the two earlier in-flight steps must NOT release the blocks:
    # their outputs don't fence the still-pending newest write.
    for out in (outs[0], outs[1]):
        scheduler.update_from_output(out, _make_model_runner_output(out))
        assert _num_draining(scheduler) > 0
        assert pool.get_num_free_blocks() == num_free_running

    # Only once the newest scheduled step's output is processed do the
    # blocks return to the pool.
    scheduler.update_from_output(outs[2], _make_model_runner_output(outs[2]))
    assert _num_draining(scheduler) == 0
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
    # A step IS globally in flight, but `short` was last scheduled in the
    # processed step, so its blocks drain immediately.
    assert scheduler.sched_step_seq > scheduler.processed_step_seq
    assert _num_draining(scheduler) == 0
    assert pool.get_num_free_blocks() > free_before  # short's blocks returned


def test_abort_mid_prefill_defers_free():
    """Intermediate prefill chunks don't allocate output placeholders, so the
    deferral must key off is_prefill_chunk: aborting a request whose prefill
    chunk is still in flight must withhold its blocks.
    """
    scheduler = create_scheduler(
        model=MODEL, async_scheduling=True, max_num_batched_tokens=16
    )
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
    assert _num_draining(scheduler) > 0
    assert pool.get_num_free_blocks() == num_free_running

    # Once the in-flight prefill step's output is processed, blocks return.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert _num_draining(scheduler) == 0
    assert pool.get_num_free_blocks() == num_free_initially


def test_non_async_abort_defers_via_last_sched_seq():
    """Without async (e.g. PP filling the pipeline) there are no placeholders
    and a full prefill isn't a partial chunk, yet an abort with a step in flight
    must defer. Only the last-scheduled-step fence catches this.

    PP=2 can't be built on a single-GPU host; the ledger needs no flag, so a
    synchronous scheduler with a step dispatched but not yet processed
    exercises it.
    """
    scheduler = create_scheduler(model=MODEL, async_scheduling=False)
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
    assert _num_draining(scheduler) > 0
    assert pool.get_num_free_blocks() == num_free_running

    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert _num_draining(scheduler) == 0
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
    pool.make_room(2)
    src_block, dst_block = pool.get_new_blocks(2)
    block_copy = KVCacheBlockCopy(
        src_block_id=src_block.block_id, dst_block_id=dst_block.block_id
    )
    manager._pending_cow_copies.append((src_block, dst_block))
    out0 = scheduler.schedule()
    assert out0.kv_cache_block_copies == [block_copy]

    # Exhaust the rest of the pool so the copy endpoints are the only blocks
    # a new request could receive, then add one that fits exactly in them.
    pool.make_room(pool.get_num_free_blocks())
    pool.get_new_blocks(pool.get_num_free_blocks())
    late_request = create_requests(
        num_requests=1,
        num_tokens=2 * scheduler.block_size,
        max_tokens=5,
        req_ids=["late"],
    )[0]
    scheduler.add_request(late_request)

    # Step 2 is scheduled while step 1 (which runs the copy) is still in
    # flight: the retentions were released stamped with the copy's step, so
    # the endpoints are DRAINING, not in the free pool -- the late request
    # must not be scheduled onto them.
    table = scheduler.kv_cache_manager.grant_table
    out1 = scheduler.schedule()
    assert src_block.ref_cnt == 0
    assert dst_block.ref_cnt == 0
    assert table.state(src_block.block_id).name == "DRAINING"
    assert table.state(dst_block.block_id).name == "DRAINING"
    assert not out1.scheduled_new_reqs

    # Step 1's output is processed: the copy has run, endpoints return to
    # the pool and the late request can be scheduled onto them safely.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert table.state(src_block.block_id).name == "FREE"
    assert table.state(dst_block.block_id).name == "FREE"
    assert _num_draining(scheduler) == 0
    out2 = scheduler.schedule()
    assert [r.req_id for r in out2.scheduled_new_reqs] == ["late"]


def _setup_async_load(defer_async_loads: bool):
    """Schedule a running request's prefill and one in-flight decode, then
    admit a request whose KV an async connector load fills.

    Returns (scheduler, importer, outputs, update_state_after_alloc mock).
    """
    scheduler = create_scheduler(
        model=MODEL,
        async_scheduling=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    scheduler.defer_async_loads = defer_async_loads
    _, out0, out1 = _setup_request_with_inflight_step(scheduler)

    importer = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, max_tokens=5, req_ids=["imp"]
    )[0]
    scheduler.add_request(importer)
    connector = scheduler.connector
    assert connector is not None
    with (
        patch.object(
            connector,
            "get_num_new_matched_tokens",
            return_value=(2 * scheduler.block_size, True),
        ),
        patch.object(connector, "update_state_after_alloc") as registered,
    ):
        out2 = scheduler.schedule()
    assert importer.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    return scheduler, importer, [out0, out1, out2], registered


def test_deferred_async_load_registers_after_inflight_step():
    """HiSparse keeps reading a released page until the pool reuses it, and an
    async connector load isn't ordered against the in-flight step, so the load
    is registered only once that step is processed.
    """
    scheduler, importer, outputs, registered = _setup_async_load(defer_async_loads=True)
    out0, out1, _ = outputs
    registered.assert_not_called()

    connector = scheduler.connector
    with patch.object(connector, "update_state_after_alloc") as registered:
        scheduler.update_from_output(out0, _make_model_runner_output(out0))
        outputs.append(scheduler.schedule())
        registered.assert_not_called()

        scheduler.update_from_output(out1, _make_model_runner_output(out1))
        outputs.append(scheduler.schedule())
    registered.assert_called_once()
    assert registered.call_args.args[0] is importer


def test_deferred_async_load_records_hit_sources_when_started():
    """The connector attributes cache-hit tiers only once the load is
    registered, so a deferred load records its prefix cache stats then."""
    scheduler, importer, outputs, _ = _setup_async_load(defer_async_loads=True)
    out0, out1, _ = outputs
    connector = scheduler.connector
    scheduler.connector_prefix_cache_stats = PrefixCacheStats()
    sources = MagicMock()
    with (
        patch.object(connector, "update_state_after_alloc"),
        patch.object(
            connector, "get_external_cache_hit_sources", return_value=sources
        ) as hit_sources,
        patch.object(PrefixCacheStats, "record") as record,
    ):
        scheduler.update_from_output(out0, _make_model_runner_output(out0))
        scheduler.update_from_output(out1, _make_model_runner_output(out1))
    hit_sources.assert_called_once()
    assert hit_sources.call_args.args[0] is importer
    assert record.call_args.kwargs["hits_by_source"] is sources


def test_async_load_registers_immediately_without_deferral():
    scheduler, importer, _, registered = _setup_async_load(defer_async_loads=False)
    registered.assert_called_once()


def test_abort_deferred_async_load_frees_after_inflight_step():
    """No load was posted for a deferred request, so nothing will report it
    done; its blocks are freed once the step that may still read them is
    processed.
    """
    scheduler, importer, outputs, _ = _setup_async_load(defer_async_loads=True)
    pool = scheduler.kv_cache_manager.block_pool
    num_free_held = pool.get_num_free_blocks()

    scheduler.finish_requests(importer.request_id, RequestStatus.FINISHED_ABORTED)
    assert importer.request_id not in scheduler.requests
    assert _num_draining(scheduler) > 0
    assert pool.get_num_free_blocks() == num_free_held

    for output in outputs:
        scheduler.update_from_output(output, _make_model_runner_output(output))
    assert _num_draining(scheduler) == 0
    assert pool.get_num_free_blocks() > num_free_held


def test_sync_preemption_frees_blocks_in_same_schedule():
    """Sync scheduling: a preempted request's last step was processed, so its
    blocks drain at once and the preempting request uses them in the same
    schedule() call (no extra step, no capacity loss)."""
    scheduler = create_scheduler(
        model=MODEL, num_blocks=11, block_size=16, max_num_batched_tokens=100
    )
    pool = scheduler.kv_cache_manager.block_pool
    first, second = create_requests(num_requests=2, num_tokens=80, block_size=16)
    scheduler.add_request(first)
    out0 = scheduler.schedule()
    scheduler.add_request(second)
    out1 = scheduler.schedule()
    for output in (out0, out1):
        scheduler.update_from_output(output, _make_model_runner_output(output))
    # Two 5-block prompts fill the 10 usable blocks; the first request's next
    # token needs a sixth.
    assert pool.get_num_free_blocks() == 0

    output = scheduler.schedule()
    assert second.status == RequestStatus.PREEMPTED
    assert first.request_id in output.num_scheduled_tokens
    assert _num_draining(scheduler) == 0


@pytest.mark.parametrize("num_spec", [0, 3])
def test_sync_preemption_count_unchanged_by_ledger(num_spec: int):
    """KV pressure under sync scheduling preempts exactly as often as before
    the step ledger (5 in this scenario, measured on the pre-ledger tree), and
    nothing is left draining."""
    from .test_async_scheduler import PipelinedEngine

    scheduler = create_scheduler(
        num_blocks=5,
        block_size=16,
        max_num_batched_tokens=512,
        num_speculative_tokens=num_spec or None,
        speculative_method="ngram_gpu" if num_spec else None,
    )
    requests = create_requests(
        num_requests=8, num_tokens=8, max_tokens=24, ignore_eos=True
    )
    pending = list(requests)
    num_preemptions = 0
    preempt = scheduler._preempt_request

    def counting_preempt(request, timestamp, **kwargs):
        nonlocal num_preemptions
        num_preemptions += 1
        return preempt(request, timestamp, **kwargs)

    scheduler._preempt_request = counting_preempt  # type: ignore[method-assign]
    for _ in range(2):
        scheduler.add_request(pending.pop(0))

    def add_request(step: int, engine: PipelinedEngine):
        if pending:
            scheduler.add_request(pending.pop(0))

    PipelinedEngine(
        scheduler,
        queue_size=1,
        accept_drafts=lambda step, req_id, n: (step + int(req_id)) % (n + 1),
    ).run(before_step=add_request)

    assert all(r.is_finished() and r.num_output_tokens == 24 for r in requests)
    assert num_preemptions == 5
    assert _num_draining(scheduler) == 0


def test_drains_do_not_stall_when_engine_idles():
    """Async, PP-deep pipeline under KV pressure: once every request finished
    and every step was processed, no block is left draining."""
    from .test_async_scheduler import PipelinedEngine, _create_async_pp_scheduler

    scheduler = _create_async_pp_scheduler(num_spec=3, pp_size=3)
    requests = create_requests(
        num_requests=8, num_tokens=8, max_tokens=24, ignore_eos=True
    )
    for request in requests:
        scheduler.add_request(request)
    PipelinedEngine(scheduler, queue_size=4).run()

    assert all(r.is_finished() for r in requests)
    assert scheduler.processed_step_seq == scheduler.sched_step_seq
    assert _num_draining(scheduler) == 0
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1


def test_retire_inflight_steps_drains_abandoned_steps():
    """Fault recovery drops in-flight steps without their outputs: retiring
    them lets their released blocks drain instead of leaking."""
    scheduler = _create_deferring_scheduler()
    pool = scheduler.kv_cache_manager.block_pool
    num_free_initially = pool.get_num_free_blocks()
    request, _, _ = _setup_request_with_inflight_step(scheduler)

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert _num_draining(scheduler) > 0
    assert scheduler._percy_rollback_marks  # the abandoned steps' marks

    scheduler.retire_inflight_steps()
    assert not scheduler._percy_rollback_marks  # dropped, never reach update
    assert scheduler.processed_step_seq == scheduler.sched_step_seq == 2
    assert _num_draining(scheduler) == 0
    assert pool.get_num_free_blocks() == num_free_initially
