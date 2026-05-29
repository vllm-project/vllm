# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for MRv2 GPUModelRunner.add_requests streaming input support."""

from unittest.mock import Mock

import pytest
import torch

from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def mock_model_runner_with_req_states():
    """Create a mock MRv2 GPUModelRunner with a real RequestState."""

    runner = Mock(spec=GPUModelRunner)
    runner.req_states = RequestState(
        max_num_reqs=10,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        num_speculative_steps=0,
        vocab_size=32000,
        device=torch.device("cpu"),
        model_dtype=torch.float32,
        cache_draft_logits=False,
    )
    runner.encoder_cache = None
    runner.model_state = Mock()
    runner.block_tables = Mock()
    runner.lora_state = Mock()
    runner.sampler = None
    runner.prompt_logprobs_worker = None
    runner.is_last_pp_rank = False

    # Mock staged writes — they use Triton kernels that require GPU
    runner.req_states.apply_staged_writes = Mock()

    # Bind the real methods to our mock
    runner._remove_request = GPUModelRunner._remove_request.__get__(runner)
    runner.add_requests = GPUModelRunner.add_requests.__get__(runner)
    return runner


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


def test_e2e_streaming_request_update_basic_flow(
    mock_model_runner_with_req_states,
):
    """Test that streaming sessions are updated correctly.

    This test validates that when a streaming session is updated with new
    prompt tokens:
    1. The old request state is removed (no free_indices leak)
    2. The new state is written with updated prefill_token_ids
    3. model_state and block_tables are re-registered for the new state
    """
    runner = mock_model_runner_with_req_states
    req_states = runner.req_states
    req_id = "streaming_req_0"
    initial_free = len(req_states.free_indices)

    # Step 1: Add initial request with 3 prompt tokens, all computed
    initial_req_data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        prefill_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=3,
        lora_request=None,
    )
    runner.add_requests(_make_scheduler_output([initial_req_data]))
    assert req_id in req_states.req_id_to_index
    assert len(req_states.free_indices) == initial_free - 1

    # Step 2: Create streaming update with extended prompt
    # The scheduler has already set prefill_token_ids to the full sequence
    # (original prompt + intermediate output + new prompt tokens)
    updated_req_data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        prefill_token_ids=[1, 2, 3, 10, 4, 5],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0, 1],),
        num_computed_tokens=4,  # 3 original prompt + 1 intermediate output
        lora_request=None,
    )
    runner.add_requests(_make_scheduler_output([updated_req_data]))

    # Step 3: Verify no free_indices leak (old slot recycled)
    assert len(req_states.free_indices) == initial_free - 1

    # Verify the request is still tracked with exactly one index
    assert req_id in req_states.req_id_to_index
    assert sum(1 for v in req_states.index_to_req_id.values() if v == req_id) == 1

    # Verify state was updated with new values
    new_idx = req_states.req_id_to_index[req_id]
    assert req_states.prompt_len.np[new_idx] == 3
    assert req_states.prefill_len.np[new_idx] == 6
    assert req_states.num_computed_prefill_tokens[new_idx] == 4

    # Verify model_state and block_tables were re-registered
    runner.model_state.add_request.assert_called_with(new_idx, updated_req_data)
    runner.block_tables.append_block_ids.assert_called_with(
        new_idx, ([0, 1],), overwrite=True
    )


def test_e2e_streaming_with_multimodal_features(
    mock_model_runner_with_req_states,
):
    """Test that streaming sessions with multimodal features are updated.

    This test validates that when a streaming session with mm features
    is updated:
    1. The old request state is removed (no free_indices leak)
    2. encoder_cache is cleaned up and re-registered with new mm_features
    3. model_state is re-registered (recomputes M-RoPE positions etc.)
    """
    runner = mock_model_runner_with_req_states
    req_states = runner.req_states
    req_id = "streaming_mm_req_0"
    initial_free = len(req_states.free_indices)

    # Enable encoder_cache for multimodal
    runner.encoder_cache = Mock()

    # Step 1: Add initial request with one audio feature
    mm_feature_1 = Mock()
    initial_req_data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2] + [0] * 10 + [3, 4],
        prefill_token_ids=[1, 2] + [0] * 10 + [3, 4],
        mm_features=[mm_feature_1],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=14,
        lora_request=None,
    )
    runner.add_requests(_make_scheduler_output([initial_req_data]))
    assert req_id in req_states.req_id_to_index

    # Reset mocks to track only the streaming update calls
    runner.encoder_cache.reset_mock()
    runner.model_state.reset_mock()

    # Step 2: Create streaming update with additional multimodal feature
    # The scheduler has folded the intermediate output (100) into
    # prefill_token_ids and added a new audio chunk
    mm_feature_2 = Mock()
    updated_req_data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2] + [0] * 10 + [3, 4],
        prefill_token_ids=[1, 2] + [0] * 10 + [3, 4, 100] + [0] * 5 + [5],
        mm_features=[mm_feature_1, mm_feature_2],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0, 1],),
        num_computed_tokens=14,
        lora_request=None,
    )
    runner.add_requests(_make_scheduler_output([updated_req_data]))

    # Step 3: Verify no free_indices leak
    assert len(req_states.free_indices) == initial_free - 1
    assert sum(1 for v in req_states.index_to_req_id.values() if v == req_id) == 1

    # Verify encoder_cache was cleaned up and re-registered
    runner.encoder_cache.remove_request.assert_called_once_with(req_id)
    runner.encoder_cache.add_request.assert_called_once_with(
        req_id, [mm_feature_1, mm_feature_2]
    )

    # Verify model_state was re-registered with new data
    new_idx = req_states.req_id_to_index[req_id]
    runner.model_state.add_request.assert_called_once_with(new_idx, updated_req_data)

    # Verify updated prefill length
    assert req_states.prefill_len.np[new_idx] == 21
