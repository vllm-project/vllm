# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SWA bounded replay: after a prefix hit the scheduler recomputes the tail of
the hit to rebuild the non-cacheable sliding-window group, keeps the hit's
blocks, and hands the worker the replay start."""

from unittest.mock import MagicMock

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
) -> Scheduler:
    """A hybrid layout: one prefix-cacheable full-attention group and one
    replayed sliding-window group."""
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
            KVCacheGroupSpec(
                ["swa"],
                SlidingWindowMLASpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.bfloat16,
                    sliding_window=WINDOW,
                    bounded_replay=True,
                ),
            ),
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
    assert scheduler.prefix_replay_tokens == WINDOW
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


def test_prefill_producer_runs_explicit_replay_before_sampling():
    scheduler = _replay_scheduler()
    scheduler.prefill_side_swa_replay = True
    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        max_tokens=1,
        block_size=BLOCK_SIZE,
    )[0]
    scheduler.add_request(request)

    prefill = scheduler.schedule()
    assert prefill.defer_prefill_sampling
    blocks_before_replay = tuple(
        len(group)
        for group in scheduler.kv_cache_manager.get_blocks(request.request_id).blocks
    )
    scheduler.update_from_output(
        prefill,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
        ),
    )

    assert request.prefill_side_replay_active
    assert not request.prefill_side_replay_done
    assert request.replay_start == NUM_PROMPT_TOKENS - WINDOW
    assert request.num_computed_tokens == NUM_PROMPT_TOKENS - WINDOW

    replay = scheduler.schedule()
    assert not replay.defer_prefill_sampling
    assert replay.num_scheduled_tokens == {request.request_id: WINDOW}
    assert replay.scheduled_cached_reqs.prefill_side_replay_starts == {
        request.request_id: NUM_PROMPT_TOKENS - WINDOW
    }
    assert (
        tuple(
            len(group)
            for group in scheduler.kv_cache_manager.get_blocks(
                request.request_id
            ).blocks
        )
        == blocks_before_replay
    )

    scheduler.update_from_output(replay, _step_output(replay, [request]))
    assert not request.prefill_side_replay_active
    assert request.prefill_side_replay_done
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


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
    assert new_req.replay_start == replay_start
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

    def full_blocks():
        return scheduler.kv_cache_manager.get_blocks(second.request_id).blocks[FULL]

    scheduled: list[int] = []
    while second.num_computed_tokens < NUM_PROMPT_TOKENS:
        out = scheduler.schedule()
        if not scheduled:
            new_req = _new_req_data(out, second)
            assert new_req.num_computed_tokens == HIT_TOKENS - WINDOW
            assert new_req.replay_start == HIT_TOKENS - WINDOW
            # A replay-only chunk adopts the hit and allocates nothing new.
            assert len(full_blocks()) == HIT_TOKENS // BLOCK_SIZE
        scheduled.append(out.num_scheduled_tokens[second.request_id])
        scheduler.update_from_output(out, _step_output(out, [first, second]))
    # 32 replayed + 4 new tokens in 16-token chunks: two replay-only chunks.
    assert scheduled == [BLOCK_SIZE, BLOCK_SIZE, NUM_PROMPT_TOKENS - HIT_TOKENS]
    assert len(full_blocks()) == -(-NUM_PROMPT_TOKENS // BLOCK_SIZE)
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
    assert new_req.replay_start == matched - WINDOW
    assert (
        out.num_scheduled_tokens[request.request_id]
        == NUM_PROMPT_TOKENS - matched + WINDOW
    )


@pytest.mark.parametrize("matched_tokens", [HIT_TOKENS - 1, HIT_TOKENS])
def test_certified_remote_checkpoint_only_recomputes_last_token(matched_tokens):
    """A complete remote replay checkpoint keeps its SWA tail."""
    prompt_tokens = HIT_TOKENS
    scheduler = _replay_scheduler(
        use_kv_connector=MockKVConfig(
            matched_tokens=matched_tokens,
            is_async=True,
            certifies_prefix_replay=True,
        )
    )
    request = create_requests(
        num_requests=1, num_tokens=prompt_tokens, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    scheduler.update_from_output(
        out, create_model_runner_output([], finished_recving={request.request_id})
    )

    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    assert new_req.replay_start == matched_tokens - WINDOW
    assert new_req.num_computed_tokens == prompt_tokens - 1
    assert out.num_scheduled_tokens[request.request_id] == 1


def test_uncertified_partial_checkpoint_falls_back_to_aligned_replay():
    matched_tokens = HIT_TOKENS - 1
    scheduler = _replay_scheduler(
        use_kv_connector=MockKVConfig(
            matched_tokens=matched_tokens,
            is_async=True,
        )
    )
    scheduler.connector.can_load_prefix_replay_checkpoint = MagicMock(return_value=True)
    request = create_requests(
        num_requests=1, num_tokens=HIT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    scheduler.update_from_output(
        out, create_model_runner_output([], finished_recving={request.request_id})
    )

    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    aligned_hit = matched_tokens // BLOCK_SIZE * BLOCK_SIZE
    assert new_req.replay_start == aligned_hit - WINDOW
    assert new_req.num_computed_tokens == aligned_hit - WINDOW


def test_remote_kv_hit_is_taken_in_whole_blocks():
    """A hit ending one token short of a block boundary would put the replay
    window's first token in a block the sliding-window group retires, so a
    connector hit is cut back to whole blocks."""
    matched = 4 * BLOCK_SIZE - 1
    scheduler = _replay_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=matched, is_async=True)
    )
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    scheduler.update_from_output(
        out, create_model_runner_output([], finished_recving={request.request_id})
    )
    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    hit = 3 * BLOCK_SIZE
    assert new_req.replay_start == hit - WINDOW
    assert new_req.num_computed_tokens == hit - WINDOW
    swa_manager = scheduler.kv_cache_manager.coordinator.single_type_managers[SWA]
    swa_blocks = scheduler.kv_cache_manager.get_blocks(request.request_id).blocks[SWA]
    assert swa_blocks[new_req.replay_start // BLOCK_SIZE] is not swa_manager._null_block


@pytest.mark.parametrize("via_connector", [False, True])
def test_hit_no_longer_than_window_is_ignored(via_connector):
    """Such a hit would be recomputed in full anyway. It is not adopted (an
    async load of it is not even started), so a request has computed tokens
    iff it replays."""
    if via_connector:
        scheduler = _replay_scheduler(
            use_kv_connector=MockKVConfig(matched_tokens=WINDOW, is_async=True)
        )
        request = create_requests(
            num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
        )[0]
    else:
        scheduler = _replay_scheduler()
        first = create_requests(
            num_requests=1, num_tokens=WINDOW, same_prompt=True, block_size=BLOCK_SIZE
        )[0]
        _prefill(scheduler, first)
        request = create_requests(
            num_requests=1,
            num_tokens=NUM_PROMPT_TOKENS,
            same_prompt=True,
            block_size=BLOCK_SIZE,
            req_ids=["long"],
        )[0]
    scheduler.add_request(request)
    out = scheduler.schedule()
    new_req = _new_req_data(out, request)
    assert request.status == RequestStatus.RUNNING
    assert new_req.num_computed_tokens == 0
    assert new_req.replay_start == 0
    assert out.num_scheduled_tokens[request.request_id] == NUM_PROMPT_TOKENS
