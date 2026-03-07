# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests that AsyncScheduler correctly handles streaming/resumable requests.

Reproduces the bug described in https://github.com/vllm-project/vllm/issues/35755
where num_output_placeholders goes negative when a streaming update arrives
during async scheduling, causing a fatal AssertionError crash.
"""

from collections import deque

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _make_resumable_request(
    request_id: str,
    num_prompt_tokens: int = 10,
    max_tokens: int = 3,
) -> Request:
    """Create a resumable (streaming) request, like a realtime ASR session."""
    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_prompt_tokens)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=block_hasher,
        resumable=True,
    )


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    """Create a ModelRunnerOutput that produces one token per scheduled req."""
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[100 + i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _make_model_runner_output_with_forced_tokens(
    scheduler_output: SchedulerOutput,
    req_ids_with_tokens: set[str],
) -> ModelRunnerOutput:
    """Create a ModelRunnerOutput where specified requests produce tokens
    even during what the scheduler considers a prefill chunk.

    This simulates encoder-decoder models (like Qwen3-ASR) where the
    encoder can finish processing and emit output tokens during prefill.
    """
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    sampled = []
    for req_id in req_ids:
        if req_id in req_ids_with_tokens:
            sampled.append([200])
        else:
            sampled.append([100])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _run_until_waiting_for_stream(scheduler, request_id: str):
    """Run the scheduler loop until a resumable request finishes its current
    segment and is waiting for more streaming input.

    Uses a single-in-flight pipeline (no pipelining) so the scheduler
    state is clean when we return.
    """
    pending: deque[SchedulerOutput] = deque()
    pending.append(scheduler.schedule())

    while pending:
        so = pending.popleft()
        mro = _make_model_runner_output(so)
        scheduler.update_from_output(so, mro)

        request = scheduler.requests.get(request_id)
        if request is None:
            break
        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            break

        so_next = scheduler.schedule()
        if so_next.num_scheduled_tokens:
            pending.append(so_next)


def test_streaming_update_placeholder_underflow():
    """Reproduce the num_output_placeholders underflow crash (issue #35755).

    The bug occurs when:
    1. A resumable request completes its first segment (prefill + decode).
    2. A streaming update arrives with new prompt tokens.
    3. The new tokens require chunked prefill (don't all fit in one step).
    4. _update_after_schedule sees is_prefill_chunk=True and skips the
       placeholder increment in AsyncScheduler.
    5. But the model output still contains a sampled token (encoder-decoder
       models like Qwen3-ASR can emit tokens during prefill).
    6. _update_request_with_output: num_output_placeholders -= 1
       → 0 - 1 = -1 → AssertionError (the bug).

    Key: max_num_batched_tokens must be small enough that the streaming
    update's new tokens require chunked prefill (not all fit in one step).
    """
    # Use a small batch budget so the streaming update tokens get chunked.
    # Initial prompt (4 tokens) fits in one step, but the streaming
    # update (10 tokens) won't → chunked prefill → is_prefill_chunk=True.
    BATCH_BUDGET = 4

    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_seqs=1,
        max_num_batched_tokens=BATCH_BUDGET,
        max_model_len=256,
        block_size=BLOCK_SIZE,
    )

    # Step 1: Create and run initial segment to completion.
    # Prompt=4 tokens fits in one schedule step with budget=4.
    req = _make_resumable_request("session-0", num_prompt_tokens=4, max_tokens=2)
    scheduler.add_request(req)

    _run_until_waiting_for_stream(scheduler, "session-0")
    assert req.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert req.num_output_placeholders == 0, (
        f"Expected placeholders=0 after completing segment, "
        f"got {req.num_output_placeholders}"
    )

    # Step 2: Submit a streaming update with more tokens than the batch
    # budget. This ensures chunked prefill → is_prefill_chunk=True.
    segment2 = _make_resumable_request("session-0", num_prompt_tokens=10, max_tokens=2)
    scheduler.add_request(segment2)
    assert req.status == RequestStatus.WAITING

    # Verify pre-conditions: new tokens exceed what was computed.
    tokens_to_compute = req.num_tokens - req.num_computed_tokens
    assert tokens_to_compute > BATCH_BUDGET, (
        f"Need chunked prefill: {tokens_to_compute} new tokens > "
        f"{BATCH_BUDGET} batch budget"
    )

    # Step 3: Schedule the request. Only BATCH_BUDGET tokens fit, so
    # is_prefill_chunk=True and AsyncScheduler skips placeholder increment.
    so = scheduler.schedule()
    assert "session-0" in so.num_scheduled_tokens
    assert req.is_prefill_chunk, (
        "Request should be in prefill chunk (not all new tokens scheduled)"
    )
    assert req.num_output_placeholders == 0, (
        f"Bug precondition: placeholders should be 0 because "
        f"AsyncScheduler skipped increment (is_prefill_chunk=True), "
        f"got {req.num_output_placeholders}"
    )

    # Step 4: Simulate an encoder-decoder model producing a token during
    # this prefill step. This is the normal behavior for models like
    # Qwen3-ASR where the encoder completes and the decoder emits tokens.
    mro = _make_model_runner_output_with_forced_tokens(
        so, req_ids_with_tokens={"session-0"}
    )

    # Step 5: In the buggy code, this would crash:
    # num_output_placeholders (0) -= len(new_token_ids) (1) = -1
    # assert num_output_placeholders >= 0  → AssertionError
    #
    # With the fix, the placeholder count is clamped to 0 instead.
    scheduler.update_from_output(so, mro)

    assert req.num_output_placeholders >= 0, (
        f"num_output_placeholders must never go negative, "
        f"got {req.num_output_placeholders}"
    )
