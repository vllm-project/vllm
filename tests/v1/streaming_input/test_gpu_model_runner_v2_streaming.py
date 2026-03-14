# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MRv2 streaming input support.

Verifies that when a streaming update arrives (same req_id via
scheduled_new_reqs), the model runner properly cleans up old state
before re-adding, preventing free_indices leaks in RequestState
and ensuring all subsystems are correctly updated.

These tests call GPUModelRunner.add_requests() directly with mock
dependencies. They should FAIL without the streaming guard in
add_requests() and PASS once it is implemented.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.cpu_test

MAX_NUM_REQS = 8
MAX_MODEL_LEN = 64


def _make_req_states():
    return RequestState(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=64,
        num_speculative_steps=0,
        vocab_size=128,
        device=torch.device("cpu"),
        model_dtype=torch.float32,
        cache_draft_logits=False,
    )


def _make_new_req_data(
    req_id,
    prompt_token_ids,
    prefill_token_ids,
    num_computed_tokens,
    sampling_params=None,
    mm_features=None,
    block_ids=None,
    lora_request=None,
):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        prefill_token_ids=prefill_token_ids,
        mm_features=mm_features or [],
        sampling_params=sampling_params,
        pooling_params=None,
        block_ids=block_ids or ([0],),
        num_computed_tokens=num_computed_tokens,
        lora_request=lora_request,
    )


def _make_scheduler_output(new_reqs):
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _make_mock_model_runner(req_states):
    """Create a mock GPUModelRunner with real RequestState.

    We mock apply_staged_writes on req_states since the Triton kernels
    inside StagedWriteTensor cannot run on CPU tensors.
    """
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = Mock(spec=GPUModelRunner)
    runner.req_states = req_states
    runner.encoder_cache = None
    runner.model_state = Mock()
    runner.block_tables = Mock()
    runner.lora_state = Mock()
    runner.sampler = None
    runner.prompt_logprobs_worker = None
    runner.is_last_pp_rank = False

    # Mock staged writes — they use Triton kernels that require GPU
    req_states.apply_staged_writes = Mock()

    # Bind the real add_requests method to our mock
    runner.add_requests = GPUModelRunner.add_requests.__get__(runner)
    return runner


# ── Test 1: Index leak prevention ──


def test_streaming_update_no_index_leak():
    """After a streaming update, only 1 free_indices slot should be used.

    Fails without the streaming guard: add_requests does not call
    remove_request before re-adding, leaking the old req_idx.
    """
    req_states = _make_req_states()
    runner = _make_mock_model_runner(req_states)
    initial_free = len(req_states.free_indices)

    # Step 1: Initial request
    runner.add_requests(
        _make_scheduler_output(
            [
                _make_new_req_data("stream-0", [1, 2, 3], [1, 2, 3], 3),
            ]
        )
    )
    assert len(req_states.free_indices) == initial_free - 1

    # Step 2: Streaming update — same req_id, extended tokens
    runner.add_requests(
        _make_scheduler_output(
            [
                _make_new_req_data("stream-0", [1, 2, 3], [1, 2, 3, 10, 4, 5], 4),
            ]
        )
    )

    # Should still only use 1 slot
    assert len(req_states.free_indices) == initial_free - 1
    # No stale entries in index_to_req_id
    assert sum(1 for v in req_states.index_to_req_id.values() if v == "stream-0") == 1


def test_repeated_streaming_updates_dont_exhaust_indices():
    """Repeated streaming updates should never exhaust free_indices.

    Fails without the streaming guard: each update leaks a slot,
    and after max_num_reqs updates we hit 'No free indices'.
    """
    req_states = _make_req_states()
    runner = _make_mock_model_runner(req_states)

    # Initial request
    runner.add_requests(
        _make_scheduler_output(
            [
                _make_new_req_data("stream-0", [1, 2], [1, 2], 2),
            ]
        )
    )

    # Simulate many streaming updates — more than max_num_reqs
    for i in range(MAX_NUM_REQS * 3):
        runner.add_requests(
            _make_scheduler_output(
                [
                    _make_new_req_data("stream-0", [1, 2], [1, 2, 10 + i], 2),
                ]
            )
        )

    # Still only 1 slot used
    assert len(req_states.free_indices) == MAX_NUM_REQS - 1
    assert "stream-0" in req_states.req_id_to_index


# ── Test 2: Subsystem cleanup ──


def test_streaming_update_cleans_up_subsystems():
    """Streaming update should call remove on encoder_cache, lora_state,
    and prompt_logprobs_worker before re-adding.

    Fails without the streaming guard: remove_request is never called
    on subsystems for the old state.
    """
    req_states = _make_req_states()
    runner = _make_mock_model_runner(req_states)
    runner.encoder_cache = Mock()
    runner.prompt_logprobs_worker = Mock()

    # Initial request
    runner.add_requests(
        _make_scheduler_output(
            [
                _make_new_req_data("stream-0", [1, 2, 3], [1, 2, 3], 3),
            ]
        )
    )

    # Reset mocks to track only the streaming update calls
    runner.encoder_cache.reset_mock()
    runner.lora_state.reset_mock()
    runner.prompt_logprobs_worker.reset_mock()

    # Streaming update
    runner.add_requests(
        _make_scheduler_output(
            [
                _make_new_req_data("stream-0", [1, 2, 3], [1, 2, 3, 10, 4, 5], 4),
            ]
        )
    )

    # Verify remove was called before re-add
    runner.encoder_cache.remove_request.assert_called_once_with("stream-0")
    runner.prompt_logprobs_worker.remove_request.assert_called_once_with("stream-0")
    runner.lora_state.remove_request.assert_called_once_with("stream-0")


# ── Test 3: Mixed batch — streaming + non-streaming ──


def test_streaming_update_with_concurrent_requests():
    """When a batch has both a streaming update and a new request,
    only the streaming request should trigger cleanup. The new
    request should be added normally.

    Fails without the streaming guard: the streaming request leaks
    an index, reducing available slots for new requests.
    """
    req_states = _make_req_states()
    runner = _make_mock_model_runner(req_states)
    initial_free = len(req_states.free_indices)

    # Add initial streaming request
    runner.add_requests(
        _make_scheduler_output(
            [
                _make_new_req_data("stream-0", [1, 2, 3], [1, 2, 3], 3),
            ]
        )
    )
    assert len(req_states.free_indices) == initial_free - 1

    # Batch with: streaming update for stream-0 + brand new request
    runner.add_requests(
        _make_scheduler_output(
            [
                _make_new_req_data("stream-0", [1, 2, 3], [1, 2, 3, 10, 4, 5], 4),
                _make_new_req_data("new-req-1", [7, 8, 9], [7, 8, 9], 0),
            ]
        )
    )

    # 2 slots used total (1 recycled streaming + 1 new)
    assert len(req_states.free_indices) == initial_free - 2
    assert "stream-0" in req_states.req_id_to_index
    assert "new-req-1" in req_states.req_id_to_index

    # Each req_id maps to exactly one index
    all_mapped_ids = list(req_states.index_to_req_id.values())
    assert all_mapped_ids.count("stream-0") == 1
    assert all_mapped_ids.count("new-req-1") == 1
