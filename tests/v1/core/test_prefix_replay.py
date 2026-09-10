# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SWA bounded replay: after a prefix-cache hit the scheduler recomputes the
trailing hit tokens without giving up their blocks, and `kv_write_start` tells
the worker where paged KV already exists."""

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

BLOCK_SIZE = 16
NUM_PROMPT_TOKENS = 100
# 100 tokens -> 6 full blocks cached -> 96-token hit.
HIT_TOKENS = NUM_PROMPT_TOKENS // BLOCK_SIZE * BLOCK_SIZE


def _model_runner_output(scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[0] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _scheduler_with_cached_prompt(replay_tokens: int):
    scheduler = create_scheduler(block_size=BLOCK_SIZE, enable_prefix_caching=True)
    scheduler.prefix_replay_tokens = replay_tokens
    first, second = create_requests(
        num_requests=2,
        num_tokens=NUM_PROMPT_TOKENS,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(first)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[first.request_id] == NUM_PROMPT_TOKENS
    scheduler.update_from_output(out, _model_runner_output(out))
    return scheduler, first, second


@pytest.mark.parametrize(
    "replay_tokens, expected_computed",
    [(0, HIT_TOKENS), (32, HIT_TOKENS - 32), (128, 0)],
)
def test_hit_is_replayed_without_reallocating(replay_tokens, expected_computed):
    scheduler, first, second = _scheduler_with_cached_prompt(replay_tokens)
    manager = scheduler.kv_cache_manager
    free_before = manager.block_pool.get_num_free_blocks()

    scheduler.add_request(second)
    out = scheduler.schedule()

    new_req = next(r for r in out.scheduled_new_reqs if r.req_id == second.request_id)
    assert new_req.num_computed_tokens == expected_computed
    assert new_req.kv_write_start == HIT_TOKENS
    assert (
        out.num_scheduled_tokens[second.request_id]
        == NUM_PROMPT_TOKENS - expected_computed
    )
    # The hit blocks are shared with the first request; only the tail block
    # beyond the hit is new. Replayed tokens never allocate.
    first_blocks = manager.get_blocks(first.request_id).get_block_ids()[0]
    second_blocks = manager.get_blocks(second.request_id).get_block_ids()[0]
    assert (
        second_blocks[: HIT_TOKENS // BLOCK_SIZE]
        == first_blocks[: HIT_TOKENS // BLOCK_SIZE]
    )
    assert free_before - manager.block_pool.get_num_free_blocks() == 1

    scheduler.update_from_output(out, _model_runner_output(out))
    assert second.num_computed_tokens == NUM_PROMPT_TOKENS


def test_replay_only_chunk_waits_for_budget():
    """A chunk holding replayed tokens only makes no progress, so the request
    waits until the prefill budget allows at least one new token."""
    scheduler, first, second = _scheduler_with_cached_prompt(replay_tokens=32)
    scheduler.scheduler_config.long_prefill_token_threshold = 32
    scheduler.add_request(second)
    out = scheduler.schedule()
    assert second.request_id not in out.num_scheduled_tokens
    assert out.num_scheduled_tokens[first.request_id] == 1
