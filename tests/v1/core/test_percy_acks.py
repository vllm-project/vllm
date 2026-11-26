# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy M2: the GrantTable learns block contents only from processed steps.

`valid` rises only on acks (I8) and a block is published to the prefix cache
only once every slot is confirmed (I3); a worker-refused row is discarded and
resynced.
"""

import pytest

from vllm.v1.core.grant_table import BlockState
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .test_async_scheduler import PipelinedEngine
from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _output(so: SchedulerOutput, tokens: dict[str, list[int]] | None = None):
    req_ids = list(so.num_scheduled_tokens)
    tokens = tokens or {}
    return ModelRunnerOutput(
        req_ids=req_ids, sampled_token_ids=[tokens.get(r, [0]) for r in req_ids]
    )


def _blocks(scheduler, request) -> list:
    return scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks.get(request.request_id, [])


def _assert_acked_only(scheduler, request) -> None:
    """Every block holds at most the slots processed steps wrote (I8), and a
    block is published only when all of its slots are confirmed (I3)."""
    table = scheduler.kv_cache_manager.grant_table
    confirmed = max(0, request.num_computed_tokens - request.num_in_flight_tokens)
    for col, block in enumerate(_blocks(scheduler, request)):
        if block.is_null:
            continue
        written = min(BLOCK_SIZE, max(0, confirmed - col * BLOCK_SIZE))
        assert table.valid(block.block_id) <= max(written, 0) or (
            table.state(block.block_id) == BlockState.SHARED
        ), (col, table.valid(block.block_id), written)
        if block.block_hash is not None:
            assert table.valid(block.block_id) == BLOCK_SIZE
            assert table.state(block.block_id) == BlockState.SHARED


def _hit_tokens(scheduler, request) -> int:
    _, num_hit, _ = scheduler.kv_cache_manager.get_computed_blocks(request)
    return num_hit


def test_block_published_only_after_its_step_is_processed():
    """A second request with the same prompt misses while the first one's
    prefill is in flight, and hits once that step is processed."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    first, second = create_requests(
        2, num_tokens=4 * BLOCK_SIZE, max_tokens=4, block_size=BLOCK_SIZE
    )
    second.prompt_token_ids[:] = first.prompt_token_ids
    second.block_hashes[:] = first.block_hashes
    scheduler.add_request(first)
    step = scheduler.schedule()
    blocks = _blocks(scheduler, first)
    table = scheduler.kv_cache_manager.grant_table

    # Planned, not acked: nothing is published, nothing is confirmed.
    assert all(b.block_hash is None for b in blocks)
    assert all(table.valid(b.block_id) == 0 for b in blocks)
    assert _hit_tokens(scheduler, second) == 0

    scheduler.update_from_output(step, _output(step))
    assert all(table.valid(b.block_id) == BLOCK_SIZE for b in blocks)
    assert all(table.state(b.block_id) == BlockState.SHARED for b in blocks)
    # The hit stops one token short of the prompt, at a block boundary.
    assert _hit_tokens(scheduler, second) == 3 * BLOCK_SIZE


def test_async_publish_waits_for_in_flight_prefill():
    """Async scheduling: the prefill is in flight while the next step is
    scheduled; its blocks stay unpublished until its output is processed."""
    scheduler = create_scheduler(
        enable_prefix_caching=True, block_size=BLOCK_SIZE, async_scheduling=True
    )
    engine = PipelinedEngine(scheduler, queue_size=2)
    first, second = create_requests(
        2, num_tokens=2 * BLOCK_SIZE + 1, max_tokens=8, block_size=BLOCK_SIZE
    )
    second.prompt_token_ids[:] = first.prompt_token_ids
    second.block_hashes[:] = first.block_hashes
    scheduler.add_request(first)
    assert engine._schedule()
    assert engine._schedule()
    assert _hit_tokens(scheduler, second) == 0
    _assert_acked_only(scheduler, first)
    engine._process_oldest_step()
    _assert_acked_only(scheduler, first)
    assert _hit_tokens(scheduler, second) == 2 * BLOCK_SIZE


@pytest.mark.parametrize("num_accepted", [0, 1, 3])
def test_spec_decode_valid_never_exceeds_accepted(num_accepted: int):
    """Rejected drafts were written into the cache but are never confirmed:
    `valid` follows the accepted prefix only (sync scheduling)."""
    num_spec = 3
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        num_speculative_tokens=num_spec,
    )
    (request,) = create_requests(
        1, num_tokens=BLOCK_SIZE - 2, max_tokens=64, block_size=BLOCK_SIZE
    )
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _output(prefill))
    for step_idx in range(4):
        scheduler.update_draft_token_ids(
            DraftTokenIds([request.request_id], [[1] * num_spec])
        )
        decode = scheduler.schedule()
        assert decode.num_scheduled_tokens[request.request_id] == 1 + num_spec
        before = request.num_computed_tokens
        tokens = list(range(10 + step_idx, 11 + step_idx + num_accepted))
        scheduler.update_from_output(
            decode, _output(decode, {request.request_id: tokens})
        )
        assert request.num_computed_tokens == before - (num_spec - num_accepted)
        _assert_acked_only(scheduler, request)


def test_async_spec_decode_valid_never_exceeds_accepted():
    """Async scheduling with rejections: drafts of in-flight steps are counted
    as computed but never confirmed until their step is processed."""
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        async_scheduling=True,
        num_speculative_tokens=2,
    )
    engine = PipelinedEngine(
        scheduler, queue_size=2, accept_drafts=lambda step, _req, n: step % 3 % (n + 1)
    )
    (request,) = create_requests(
        1, num_tokens=BLOCK_SIZE + 3, max_tokens=60, block_size=BLOCK_SIZE
    )
    scheduler.add_request(request)
    for _ in range(30):
        if request.is_finished():
            break
        if len(engine.queue) < engine.queue_size:
            engine._schedule()
            for so, _ in list(engine.queue)[:1]:
                if so.num_scheduled_tokens.get(request.request_id):
                    scheduler.update_draft_token_ids(
                        DraftTokenIds([request.request_id], [[1, 1]])
                    )
        else:
            engine._process_oldest_step()
        _assert_acked_only(scheduler, request)


@pytest.mark.parametrize("queue_size", [2, 3, 4])
def test_deep_async_spec_pipeline_keeps_publishing(caplog_vllm, queue_size: int):
    """Async scheduling with several steps in flight and rejected drafts: a
    step's scheduled start counts the drafts of every earlier in-flight step
    as accepted, but the worker writes from the corrected position. Acks
    follow the worker-observed start exactly, so the frontier check never
    fires, blocks keep being published, and `valid` never exceeds the
    accepted tokens."""
    from .test_async_scheduler import _create_async_pp_scheduler

    scheduler = _create_async_pp_scheduler(
        3, pp_size=1, num_blocks=64, enable_prefix_caching=True
    )
    (request,) = create_requests(
        1, num_tokens=BLOCK_SIZE + 4, max_tokens=60, block_size=BLOCK_SIZE
    )
    scheduler.add_request(request)
    engine = PipelinedEngine(
        scheduler,
        queue_size=queue_size,
        # Reject some, accept others: 0, 1, 2, 3, 0, ...
        accept_drafts=lambda step, _req, n: step % (n + 1),
    )

    def check(_i, _engine):
        if request.request_id in scheduler.requests:
            _assert_acked_only(scheduler, request)

    with caplog_vllm.at_level("WARNING"):
        engine.run(before_step=check)
    assert request.num_output_tokens == 60
    assert "past its confirmed KV prefix" not in caplog_vllm.text
    pool = scheduler.kv_cache_manager.block_pool
    published = [b for b in pool.blocks if b.block_hash is not None]
    # Every full block of the request's confirmed prefix was published.
    assert len(published) >= (BLOCK_SIZE + 4 + 60 - 1) // BLOCK_SIZE - 1
    assert all(pool.grant_table.valid(b.block_id) == BLOCK_SIZE for b in published)


def test_row_check_failure_discards_output_and_resyncs():
    """A worker-refused row (W1) is not acked: its output is dropped and the
    request is preempted to be re-sent in full, while others proceed."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    refused, ok = create_requests(
        2, num_tokens=2 * BLOCK_SIZE, max_tokens=8, block_size=BLOCK_SIZE
    )
    scheduler.add_request(refused)
    scheduler.add_request(ok)
    step = scheduler.schedule()
    refused_blocks = list(_blocks(scheduler, refused))
    output = _output(step, {refused.request_id: [7], ok.request_id: [8]})
    output.row_check_failures = {refused.request_id}
    scheduler.update_from_output(step, output)

    assert refused.status == RequestStatus.PREEMPTED
    assert refused.num_output_tokens == 0
    assert refused.num_computed_tokens == 0
    assert refused not in scheduler.running
    table = scheduler.kv_cache_manager.grant_table
    assert all(b.block_hash is None for b in refused_blocks)
    assert all(
        table.state(b.block_id) in (BlockState.DRAINING, BlockState.FREE)
        for b in refused_blocks
    )
    assert ok.num_output_tokens == 1 and ok in scheduler.running

    resync = scheduler.schedule()
    assert refused.request_id in [r.req_id for r in resync.scheduled_new_reqs]


def test_cow_destination_published_only_after_copy_step():
    """`move_block_hashes` re-points a published entry to a copy: the copy is
    filled by the step being scheduled, so its entry waits for that step."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    (request,) = create_requests(
        1, num_tokens=2 * BLOCK_SIZE + 1, max_tokens=4, block_size=BLOCK_SIZE
    )
    scheduler.add_request(request)
    step = scheduler.schedule()
    scheduler.update_from_output(step, _output(step))
    pool = scheduler.kv_cache_manager.block_pool
    src = _blocks(scheduler, request)[0]
    key = src.block_hash
    assert key is not None and pool.grant_table.state(src.block_id) == (
        BlockState.SHARED
    )
    (dst,) = pool.get_new_blocks(1)
    copy_seq = pool.grant_table.latest_sched_seq + 1
    pool.move_block_hashes(src, dst)

    assert src.block_hash is None and dst.block_hash is None
    assert pool.get_cached_block(_hash_of(key), [0]) is None
    pool.drain(copy_seq - 1)
    assert dst.block_hash is None
    pool.grant_table.begin_step(copy_seq)
    pool.drain(copy_seq)
    assert dst.block_hash == key
    assert pool.grant_table.state(dst.block_id) == BlockState.SHARED
    assert pool.grant_table.valid(dst.block_id) == BLOCK_SIZE


def _hash_of(block_hash_with_group_id):
    from vllm.v1.core.kv_cache_utils import get_block_hash

    return get_block_hash(block_hash_with_group_id)


def test_replay_58259_resumable_handoff_never_publishes_unwritten_slots(
    monkeypatch, caplog_vllm
):
    """Replay of vLLM #58259 (B3; fix 2f4a9759, regression window ending at
    764413559): a resumable request whose turn stops while async steps are in
    flight resumed from the in-flight frontier instead of the materialized
    one, so the next turn treated positions no processed step wrote for it as
    computed, and vLLM published (then wrote) those blocks.

    The fix is reverted by restoring the pre-fix frontier once the
    continuation is merged. Percy refuses to confirm a step that starts past
    the confirmed prefix, so no block holding those slots is ever published
    (I3, I8) and no planned write lands in a published block (I2).
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    fixed = Scheduler._handle_stopped_request
    in_flight_frontier: dict[str, int] = {}

    def record_frontier(self, request):
        in_flight_frontier[request.request_id] = request.num_computed_tokens
        return fixed(self, request)

    monkeypatch.setattr(Scheduler, "_handle_stopped_request", record_frontier)
    scheduler = create_scheduler(
        async_scheduling=True, enable_prefix_caching=True, block_size=BLOCK_SIZE
    )
    engine = PipelinedEngine(scheduler, queue_size=3)
    engine._next_token = EOS_TOKEN_ID
    request_id = "resumable-race"
    (request,) = create_requests(
        1,
        num_tokens=BLOCK_SIZE - 2,
        max_tokens=4,
        req_ids=[request_id],
        block_size=BLOCK_SIZE,
    )
    request.resumable = True
    (continuation,) = create_requests(
        1,
        num_tokens=3 * BLOCK_SIZE,
        max_tokens=4,
        req_ids=[request_id],
        block_size=BLOCK_SIZE,
    )
    continuation.resumable = True
    scheduler.add_request(request)
    for _ in range(3):
        assert engine._schedule()
    engine._process_oldest_step()
    scheduler.add_request(continuation)
    # Revert the fix: resume from the in-flight frontier.
    safe_frontier = request.num_computed_tokens
    request.num_computed_tokens = in_flight_frontier[request_id]
    assert request.num_computed_tokens > safe_frontier

    with caplog_vllm.at_level("WARNING"):
        for _ in range(16):
            if engine.queue and (
                len(engine.queue) >= 3 or not scheduler.has_requests()
            ):
                engine._process_oldest_step()
            elif scheduler.has_requests():
                engine._schedule()
            else:
                break
            if request_id in scheduler.requests:
                _assert_acked_only(scheduler, request)
    assert "past its confirmed KV prefix" in caplog_vllm.text
    assert "check_writes" not in caplog_vllm.text
    pool = scheduler.kv_cache_manager.block_pool
    first_unwritten_col = safe_frontier // BLOCK_SIZE
    for block in pool.blocks:
        if block.block_hash is not None:
            assert pool.grant_table.valid(block.block_id) == BLOCK_SIZE
    # Nothing at or past the unwritten gap was published for the request.
    for col, block in enumerate(_blocks(scheduler, request)):
        if col >= first_unwritten_col:
            assert block.block_hash is None


def test_hash_helper_roundtrip():
    key = make_block_hash_with_group_id(b"abc", 0)  # type: ignore[arg-type]
    assert _hash_of(key) == b"abc"
