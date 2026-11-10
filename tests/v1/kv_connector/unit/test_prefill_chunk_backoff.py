# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `SchedulerConfig.prefill_chunk_backoff`."""

import pytest

from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from .utils import (
    assert_scheduler_empty,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
CHUNK_TOKENS = BLOCK_SIZE * 16


def _make_scheduler(num_blocks: int, prefill_chunk_backoff: bool):
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE, max_num_batched_tokens=CHUNK_TOKENS
    )
    vllm_config.scheduler_config.prefill_chunk_backoff = prefill_chunk_backoff
    return create_scheduler(
        vllm_config,
        num_blocks=num_blocks,
        kv_cache_config=make_kv_cache_config(
            BLOCK_SIZE,
            swa_enabled=True,
            sw_size=BLOCK_SIZE * 2,
            num_blocks=num_blocks,
        ),
    )


def _step(scheduler, finished_sending: set[str] | None = None) -> dict[str, int]:
    """Run one engine step; a prefill samples a token only on its last chunk."""
    scheduler_output = scheduler.schedule()
    req_ids = list(scheduler_output.num_scheduled_tokens)
    sampled_token_ids = []
    for req_id in req_ids:
        req = scheduler.requests[req_id]
        prefill_done = req.num_computed_tokens >= req.num_tokens
        sampled_token_ids.append([0] if prefill_done else [])
    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled_token_ids,
            kv_connector_output=(
                KVConnectorOutput(finished_sending=finished_sending)
                if finished_sending
                else None
            ),
        ),
    )
    return dict(scheduler_output.num_scheduled_tokens)


@pytest.mark.parametrize("prefill_chunk_backoff", [False, True])
def test_prefill_chunk_backoff_when_idle(prefill_chunk_backoff):
    """KV held outside admission reservations (here: a finished P/D producer
    request whose KV the decode side has not pulled yet) leaves too few free
    blocks for the head's full-sequence check at a full chunk. Nothing runs,
    so without the backoff the engine idles until the KV is released; with it
    the head finishes its prefill at half-size chunks, without preemption.
    """
    # Full attention + sliding window (32 tokens). Admitting the 1024-token
    # head needs 64 + 19 blocks at 256-token chunks but 64 + 11 at 128-token
    # chunks; 77 of the 111 usable blocks are free while `held` is held.
    scheduler = _make_scheduler(112, prefill_chunk_backoff)
    held = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=BLOCK_SIZE * 32,
        do_remote_decode=True,
    )
    scheduler.add_request(held)
    while not held.is_finished():
        _step(scheduler)
    head = create_request(
        request_id=2,
        block_size=BLOCK_SIZE,
        num_tokens=BLOCK_SIZE * 64,
        do_remote_decode=True,
    )
    scheduler.add_request(head)

    chunks = [_step(scheduler).get(head.request_id, 0) for _ in range(20)]

    if not prefill_chunk_backoff:
        assert not any(chunks)
        return
    assert head.is_finished()
    assert sum(chunks) == head.num_prompt_tokens
    assert max(chunks) == CHUNK_TOKENS // 2
    assert head.num_preemptions == 0
    assert scheduler.num_prefill_chunk_backoffs == 1

    # The decode side pulls the KV: the chunk cap goes with the blocks.
    _step(scheduler, finished_sending={held.request_id, head.request_id})
    assert_scheduler_empty(scheduler)
    kv_cache_manager = scheduler.kv_cache_manager
    assert not kv_cache_manager.prefill_chunk_caps
    assert not any(
        manager.admission_cap_overrides
        for manager in kv_cache_manager.coordinator.single_type_managers
    )


@pytest.mark.parametrize("num_blocks", [100, 1000], ids=["tight", "roomy"])
def test_prefill_chunk_backoff_noop_when_not_idle(num_blocks):
    """Steps that schedule something are scheduled exactly as without the
    backoff, including steps where a waiting request does not fit."""
    traces = []
    for prefill_chunk_backoff in (False, True):
        scheduler = _make_scheduler(num_blocks, prefill_chunk_backoff)
        reqs = [
            create_request(
                request_id=i,
                block_size=BLOCK_SIZE,
                num_tokens=num_tokens,
                do_remote_decode=i % 2 == 0,
            )
            for i, num_tokens in enumerate((700, 300, 1000, 200, 500, 64), 1)
        ]
        for req in reqs:
            scheduler.add_request(req)
        trace = []
        finished_sending: set[str] = set()
        for _ in range(60):
            trace.append(_step(scheduler, finished_sending))
            # The decode side pulls each finished request's KV a step later.
            finished_sending = {
                req.request_id
                for req in reqs
                if req.is_finished() and req.request_id in scheduler.requests
            }
        assert all(req.is_finished() for req in reqs)
        assert scheduler.num_prefill_chunk_backoffs == 0
        traces.append(trace)
    assert traces[0] == traces[1]
