# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for MRv2 GPUModelRunner.add_requests streaming input support."""

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def _make_scheduler_output(new_reqs):
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def test_e2e_streaming_request_update_basic_flow(
    gpu_model_runner_for_requests,
):
    """Test that streaming sessions are updated correctly.

    This test validates that when a streaming session is updated with new
    prompt tokens:
    1. The old request state is removed (no free_indices leak)
    2. The new state is written with updated prefill_token_ids
    3. block tables contain the new blocks
    """
    runner = gpu_model_runner_for_requests
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

    runner.block_tables.apply_staged_writes()
    torch.testing.assert_close(
        runner.block_tables.block_tables[0].gpu[new_idx, :2].cpu(),
        torch.tensor([0, 1], dtype=torch.int32),
    )


def test_e2e_streaming_with_multimodal_features(
    gpu_model_runner_for_requests,
):
    """Test that streaming sessions with multimodal features are updated.

    This test validates that when a streaming session with mm features
    is updated:
    1. The old request state is removed (no free_indices leak)
    2. encoder_cache is cleaned up and re-registered with new mm_features
    3. the encoder cache holds all features from the updated request
    """
    runner = gpu_model_runner_for_requests
    req_states = runner.req_states
    req_id = "streaming_mm_req_0"
    initial_free = len(req_states.free_indices)

    # Enable encoder_cache for multimodal
    runner.encoder_cache = EncoderCache()

    # Step 1: Add initial request with one audio feature
    mm_feature_1 = MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier="audio-1",
        mm_position=PlaceholderRange(offset=2, length=10),
    )
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

    assert runner.encoder_cache.mm_features[req_id] == [mm_feature_1]

    # Step 2: Create streaming update with additional multimodal feature
    # The scheduler has folded the intermediate output (100) into
    # prefill_token_ids and added a new audio chunk
    mm_feature_2 = MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier="audio-2",
        mm_position=PlaceholderRange(offset=15, length=5),
    )
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

    assert runner.encoder_cache.mm_features[req_id] == [mm_feature_1, mm_feature_2]
    new_idx = req_states.req_id_to_index[req_id]

    # Verify updated prefill length
    assert req_states.prefill_len.np[new_idx] == 21
