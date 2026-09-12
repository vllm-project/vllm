# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SWA bounded replay: after a prefix hit the scheduler recomputes the tail of
the hit to rebuild the non-cacheable sliding-window group, keeps the hit's
blocks, and hands the worker the replayed range [replay_start, replay_end)."""

import pytest
import torch

from tests.v1.kv_connector.unit.utils import MockKVConfig, create_model_runner_output
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests, create_scheduler

BLOCK_SIZE = 16
WINDOW = 32
NUM_PROMPT_TOKENS = 100
# 100 tokens -> 6 full blocks cached -> 96-token hit.
HIT_TOKENS = NUM_PROMPT_TOKENS // BLOCK_SIZE * BLOCK_SIZE
FULL, SWA = 0, 1


def _replay_scheduler(
    *,
    long_prefill_token_threshold: int = 0,
    use_kv_connector: MockKVConfig | None = None,
    windows: tuple[int, ...] = (WINDOW,),
) -> Scheduler:
    """A hybrid layout: one prefix-cacheable full-attention group and one
    replayed sliding-window group per window."""
    base = create_scheduler(
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        long_prefill_token_threshold=long_prefill_token_threshold,
        use_kv_connector=use_kv_connector,
    )
    vllm_config = base.vllm_config
    kv_cache_config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
        ]
        + [
            KVCacheGroupSpec(
                [f"swa{window}"],
                SlidingWindowMLASpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.bfloat16,
                    sliding_window=window,
                    bounded_replay=True,
                ),
            )
            for window in windows
        ],
    )
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    scheduler.use_v2_model_runner = True
    assert scheduler.prefix_replay_spec is not None
    assert scheduler.prefix_replay_spec.prefix_replay_tokens == WINDOW
    return scheduler


def _prefill(scheduler: Scheduler, request) -> None:
    scheduler.add_request(request)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[request.request_id] == request.num_prompt_tokens
    scheduler.update_from_output(out, _step_output(out, [request]))


def _step_output(out, requests) -> ModelRunnerOutput:
    """Model output for every scheduled request: a token once its prefill is
    done (non-zero, so outputs never extend the all-zero shared prompt in the
    cache), nothing while it is still prefilling. schedule() has already
    advanced num_computed_tokens past this step's chunk."""
    scheduled = [r for r in requests if r.request_id in out.num_scheduled_tokens]
    return ModelRunnerOutput(
        req_ids=[r.request_id for r in scheduled],
        req_id_to_index={r.request_id: i for i, r in enumerate(scheduled)},
        sampled_token_ids=[
            [1000] if r.num_computed_tokens >= r.num_prompt_tokens else []
            for r in scheduled
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
    )


def _new_req_data(out, request):
    return next(r for r in out.scheduled_new_reqs if r.req_id == request.request_id)


def test_hit_replays_window_without_reallocating():
    scheduler = _replay_scheduler()
    expected_replay = WINDOW
    first, second = create_requests(
        num_requests=2,
        num_tokens=NUM_PROMPT_TOKENS,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )
    _prefill(scheduler, first)
    manager = scheduler.kv_cache_manager

    scheduler.add_request(second)
    out = scheduler.schedule()

    replay_start = HIT_TOKENS - expected_replay
    new_req = _new_req_data(out, second)
    assert new_req.num_computed_tokens == replay_start
    assert (new_req.replay_start, new_req.replay_end) == (replay_start, HIT_TOKENS)
    assert (
        out.num_scheduled_tokens[second.request_id] == NUM_PROMPT_TOKENS - replay_start
    )

    first_blocks = manager.get_blocks(first.request_id).blocks
    second_blocks = manager.get_blocks(second.request_id).blocks
    # The full-attention hit is shared (replayed tokens included); only the
    # tail block is new.
    num_hit_blocks = HIT_TOKENS // BLOCK_SIZE
    assert second_blocks[FULL][:num_hit_blocks] == first_blocks[FULL][:num_hit_blocks]
    assert all(b.ref_cnt == 2 for b in second_blocks[FULL][:num_hit_blocks])
    assert len(second_blocks[FULL]) == num_hit_blocks + 1
    # The window group never hits: blocks below the window are null, the
    # window (replayed and new tokens) gets fresh blocks.
    num_skipped_blocks = (HIT_TOKENS - WINDOW + 1) // BLOCK_SIZE
    assert all(b.is_null for b in second_blocks[SWA][:num_skipped_blocks])
    num_window_blocks = len(second_blocks[SWA]) - num_skipped_blocks
    assert num_window_blocks == -(-NUM_PROMPT_TOKENS // BLOCK_SIZE) - num_skipped_blocks
    assert not any(b.is_null for b in second_blocks[SWA][num_skipped_blocks:])

    scheduler.update_from_output(out, _step_output(out, [first, second]))
    assert second.num_computed_tokens == NUM_PROMPT_TOKENS


def test_replay_only_chunks_make_progress():
    """A chunk cap smaller than the replay window schedules replay-only
    chunks; they advance the request instead of stalling it."""
    scheduler = _replay_scheduler(long_prefill_token_threshold=BLOCK_SIZE)
    first, second = create_requests(
        num_requests=2,
        num_tokens=NUM_PROMPT_TOKENS,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(first)
    while first.num_computed_tokens < NUM_PROMPT_TOKENS:
        out = scheduler.schedule()
        scheduler.update_from_output(out, _step_output(out, [first]))

    scheduler.add_request(second)
    scheduled: list[int] = []
    while second.num_computed_tokens < NUM_PROMPT_TOKENS:
        out = scheduler.schedule()
        if not scheduled:
            new_req = _new_req_data(out, second)
            assert new_req.num_computed_tokens == HIT_TOKENS - WINDOW
            assert new_req.replay_end == HIT_TOKENS
        scheduled.append(out.num_scheduled_tokens[second.request_id])
        scheduler.update_from_output(out, _step_output(out, [first, second]))
    # 32 replayed + 4 new tokens in 16-token chunks: two replay-only chunks.
    assert scheduled == [BLOCK_SIZE, BLOCK_SIZE, NUM_PROMPT_TOKENS - HIT_TOKENS]
    assert second.status == RequestStatus.RUNNING


def test_async_remote_kv_hit_replays_after_load():
    """A KV-connector hit loaded asynchronously carries no window state
    either; the replay is applied when the load completes."""
    matched = 64
    scheduler = _replay_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=matched, is_async=True)
    )
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.request_id not in out.num_scheduled_tokens
    scheduler.update_from_output(
        out, create_model_runner_output([], finished_recving={request.request_id})
    )

    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    assert new_req.num_computed_tokens == matched - WINDOW
    assert (new_req.replay_start, new_req.replay_end) == (matched - WINDOW, matched)
    assert (
        out.num_scheduled_tokens[request.request_id]
        == NUM_PROMPT_TOKENS - matched + WINDOW
    )


def test_sync_remote_kv_hit_replays_like_a_local_hit():
    """A synchronously loaded KV-connector hit is admitted through the same
    lookup as a local hit and replays its last window."""
    matched = 64
    scheduler = _replay_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=matched, is_async=False)
    )
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    assert new_req.num_computed_tokens == matched - WINDOW
    assert (new_req.replay_start, new_req.replay_end) == (matched - WINDOW, matched)
    assert (
        out.num_scheduled_tokens[request.request_id]
        == NUM_PROMPT_TOKENS - matched + WINDOW
    )


def test_hit_shorter_than_window_replays_whole_hit():
    short_hit = BLOCK_SIZE  # one block, less than WINDOW
    scheduler = _replay_scheduler()
    first = create_requests(
        num_requests=1, num_tokens=short_hit, same_prompt=True, block_size=BLOCK_SIZE
    )[0]
    _prefill(scheduler, first)
    second = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        same_prompt=True,
        block_size=BLOCK_SIZE,
        req_ids=["long"],
    )[0]
    scheduler.add_request(second)
    out = scheduler.schedule()
    new_req = _new_req_data(out, second)
    assert new_req.num_computed_tokens == 0
    assert (new_req.replay_start, new_req.replay_end) == (0, short_hit)
    assert out.num_scheduled_tokens[second.request_id] == NUM_PROMPT_TOKENS


def test_chunk_ending_at_the_hit_allocates_nothing():
    """A chunk that stops exactly at the hit is pure replay: it needs no new
    slots yet must still be admitted."""
    scheduler = _replay_scheduler(long_prefill_token_threshold=WINDOW)
    first, second = create_requests(
        num_requests=2,
        num_tokens=NUM_PROMPT_TOKENS,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(first)
    while first.num_computed_tokens < NUM_PROMPT_TOKENS:
        out = scheduler.schedule()
        scheduler.update_from_output(out, _step_output(out, [first]))

    scheduler.add_request(second)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[second.request_id] == WINDOW
    num_blocks_before = len(
        scheduler.kv_cache_manager.get_blocks(second.request_id).blocks[FULL]
    )
    scheduler.update_from_output(out, _step_output(out, [first, second]))
    assert second.num_computed_tokens == HIT_TOKENS
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[second.request_id] == NUM_PROMPT_TOKENS - HIT_TOKENS
    assert (
        len(scheduler.kv_cache_manager.get_blocks(second.request_id).blocks[FULL])
        == num_blocks_before + 1
    )


def test_replay_windows_must_agree():
    with pytest.raises(AssertionError, match="replay windows"):
        _replay_scheduler(windows=(WINDOW, WINDOW // 2))


def _failed_block(scheduler: Scheduler, request, block_idx: int) -> int:
    """Block id of one of the request's loaded full-attention blocks."""
    blocks = scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[FULL]
    return blocks[block_idx].block_id


def test_sync_kv_load_failure_readmits_the_request():
    """A failed sync load cannot rewind a hybrid request in place (the window
    group holds no blocks below the replayed range), so the request starts
    over and its next admission hits the prefix that did load."""
    matched = 64
    scheduler = _replay_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=matched, is_async=False)
    )
    scheduler.recompute_kv_load_failures = True
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    assert _new_req_data(out, request).replay_end == matched

    # The block holding positions [32, 48) failed to load.
    failed = _failed_block(scheduler, request, 2)
    scheduler.update_from_output(
        out, create_model_runner_output([request], invalid_block_ids={failed})
    )
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert not scheduler.running

    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    # [0, 32) stayed cached and is hit again (the failed block was evicted);
    # the connector offers `matched` more, and the hit replays its last window.
    hit = 2 * BLOCK_SIZE + matched
    assert new_req.num_computed_tokens == hit - WINDOW
    assert (new_req.replay_start, new_req.replay_end) == (hit - WINDOW, hit)
    assert out.num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS - (
        hit - WINDOW
    )


def test_async_kv_load_failure_readmits_the_request():
    """A partially failed async load keeps its valid prefix in the cache and
    re-admits the request through the prefix-cache lookup."""
    matched = 64
    scheduler = _replay_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=matched, is_async=True)
    )
    scheduler.recompute_kv_load_failures = True
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    failed = _failed_block(scheduler, request, 2)
    scheduler.update_from_output(
        out,
        create_model_runner_output(
            [], finished_recving={request.request_id}, invalid_block_ids={failed}
        ),
    )
    assert request.num_computed_tokens == 2 * BLOCK_SIZE

    # Re-admission: the valid prefix is hit locally, the rest is loaded again.
    out = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(
        out, create_model_runner_output([], finished_recving={request.request_id})
    )
    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    hit = 2 * BLOCK_SIZE + matched
    assert new_req.num_computed_tokens == hit - WINDOW
    assert (new_req.replay_start, new_req.replay_end) == (hit - WINDOW, hit)
