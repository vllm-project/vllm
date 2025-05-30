# SPDX-License-Identifier: Apache-2.0
import unittest.mock as mock

import pytest

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.worker.tpu_model_runner import (
    TPUModelRunner, _get_padded_num_reqs_with_upper_limit,
    _get_padded_token_len, _get_req_paddings, _get_token_paddings)

# Mock torch_xla module since it may not be available in the test environments
torch_xla_patcher = mock.patch.dict(
    "sys.modules", {
        "torch_xla": mock.MagicMock(),
        "torch_xla.core.xla_model": mock.MagicMock(),
        "torch_xla.runtime": mock.MagicMock(),
    })
torch_xla_patcher.start()

# Mock the PallasAttentionBackend
pallas_attention_backend_patcher = mock.patch(
    "vllm.v1.worker.tpu_model_runner.PallasAttentionBackend", )
pallas_attention_backend_patcher.start()


@pytest.fixture
def model_runner():
    # Patchers have already been started at module level.
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        task="generate",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",  # TPUs typically use bfloat16
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
    )
    device = "xla:0"  # Mocking TPU device
    with mock.patch("vllm.v1.worker.tpu_model_runner.torch"), \
         mock.patch("vllm.v1.worker.tpu_model_runner.xm"), \
         mock.patch("vllm.v1.worker.tpu_model_runner.xr"):
        return TPUModelRunner(vllm_config, device)


@pytest.fixture(autouse=True, scope="session")
def cleanup_patches():
    yield
    torch_xla_patcher.stop()
    pallas_attention_backend_patcher.stop()


def _schedule_new_request(*req_ids: str) -> SchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req_id in req_ids:
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3],
                mm_inputs=[],
                mm_hashes=[],
                mm_positions=[],
                sampling_params=SamplingParams(),
                block_ids=[[0]],  # block_ids should be list[list[int]]
                num_computed_tokens=0,
                lora_request=None,
            ))
        num_scheduled_tokens[req_id] = 3
        total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=[],
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )


def _is_req_scheduled(model_runner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_req_state_block_table_match(model_runner, req_id: str) -> bool:
    """Check if the request state block IDs match the block table.
    
    This function handles both legacy BlockTable and new MultiGroupBlockTable
    structures for backward compatibility.
    """

    req_index = model_runner.input_batch.req_id_to_index[req_id]
    multi_group_block_table = model_runner.input_batch.block_table
    req_state = model_runner.requests[req_id]

    # Access the first block table from MultiGroupBlockTable
    # This is safe since we currently only use single KV cache groups
    block_table = multi_group_block_table[0]

    # req_state.block_ids is now list[list[int]] for MultiGroupBlockTable
    # Extract the first group's block IDs
    if isinstance(req_state.block_ids[0], list):
        # New format: list[list[int]] - extract first group
        req_block_ids = req_state.block_ids[0]
    else:
        # Legacy format: list[int] - use directly
        req_block_ids = req_state.block_ids

    if block_table.num_blocks_per_row[req_index] != len(req_block_ids):
        return False

    num_blocks = block_table.num_blocks_per_row[req_index]
    block_table_values = block_table.block_table_np[req_index, :num_blocks]
    return (block_table_values == req_block_ids).all()


def test_update_states_new_request(model_runner):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_finished(model_runner):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # finish req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids={req_id},
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    model_runner._update_states(scheduler_output)
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)


def test_update_states_request_resumed(model_runner):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # unschedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    # resume req
    cached_req_data = CachedRequestData(
        req_id=req_id,
        resumed_from_preemption=False,
        new_token_ids=[],
        new_block_ids=[],
        num_computed_tokens=0,
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[cached_req_data],
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_no_changes(model_runner):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # schedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_unscheduled(model_runner):
    req_ids = ("req_0", "req_1")

    # new reqs
    scheduler_output = _schedule_new_request(*req_ids)

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert _is_req_scheduled(model_runner, req_ids[1])

    # unschedule req_1
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={req_ids[0]: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert not _is_req_scheduled(model_runner, req_ids[1])


def test_get_paddings():
    # Bucketed padding
    min_token_size, max_token_size, padding_gap = 16, 512, 64
    expected_paddings = [16, 32, 64, 128, 192, 256, 320, 384, 448, 512]
    actual_paddings = _get_token_paddings(min_token_size, max_token_size,
                                          padding_gap)

    # Bucketed padding with max_token_size not a power of two.
    max_token_size = 317
    expected_paddings = [16, 32, 64, 128, 192, 256, 320]
    actual_paddings = _get_token_paddings(min_token_size, max_token_size,
                                          padding_gap)
    assert actual_paddings == expected_paddings

    # Exponential padding.
    max_token_size, padding_gap = 1024, 0
    expected_paddings = [16, 32, 64, 128, 256, 512, 1024]
    actual_paddings = _get_token_paddings(min_token_size, max_token_size,
                                          padding_gap)
    assert actual_paddings == expected_paddings
    # Exponential padding with max_token_size not a power of two.
    max_token_size = 317
    expected_paddings = [16, 32, 64, 128, 256, 512]
    actual_paddings = _get_token_paddings(min_token_size, max_token_size,
                                          padding_gap)
    assert actual_paddings == expected_paddings


def test_get_padded_token_len():
    min_token_size, max_token_size, padding_gap = 16, 512, 64
    paddings = _get_token_paddings(min_token_size, max_token_size, padding_gap)
    assert _get_padded_token_len(paddings, 1) == 16
    assert _get_padded_token_len(paddings, 16) == 16
    assert _get_padded_token_len(paddings, 20) == 32
    assert _get_padded_token_len(paddings, 300) == 320
    assert _get_padded_token_len(paddings, 512) == 512


def test_get_padded_num_reqs_with_upper_limit():
    assert _get_padded_num_reqs_with_upper_limit(3, 32) == 8
    assert _get_padded_num_reqs_with_upper_limit(9, 32) == 16
    assert _get_padded_num_reqs_with_upper_limit(19, 32) == 32
    assert _get_padded_num_reqs_with_upper_limit(17, 28) == 28


def test_get_req_paddings():
    assert _get_req_paddings(1, 32) == [8, 16, 32]
    assert _get_req_paddings(8, 32) == [8, 16, 32]
    assert _get_req_paddings(8, 36) == [8, 16, 32, 36]
