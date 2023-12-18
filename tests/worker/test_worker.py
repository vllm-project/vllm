
# pylint: disable=protected-access
import math
import random

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.worker.worker import Worker

from .utils import (create_execute_model_data, create_worker,
                    create_seq_group_metadata_from_prompts)


@torch.inference_mode()
def test_prepare_inputs_can_save_multiple_tokens_per_sequence():
    """Verify prepare_inputs correctly encodes input such that
    the model forward pass will save >1 token from the previous
    iteration in the KV cache.

    This mocks out the actual model call.
    """
    seed = 100
    block_size = 32
    num_gpu_blocks = 2048 // block_size
    worker = create_worker(Worker,
                           model_name='JackFram/llama-68m',
                           seed=seed,
                           block_size=block_size,
                           num_gpu_blocks=num_gpu_blocks)

    prompts = [list(range(4)), list(range(10))]
    prev_output_tokens = [list(range(2)), list(range(5))]
    num_tokens_processed = [len(prompt) + 1 for prompt in prompts]
    final_seq_lens = [
        len(prompt + output_tokens) + 1
        for prompt, output_tokens in zip(prompts, prev_output_tokens)
    ]
    num_missing_from_kv_cache = [
        len(prompt) + len(output_tokens) - num_processed
        for prompt, output_tokens, num_processed in zip(
            prompts, prev_output_tokens, num_tokens_processed)
    ]

    print(f'{prompts=}')
    print(f'{prev_output_tokens=}')
    print(f'{num_missing_from_kv_cache=}')

    execute_model_data = create_execute_model_data(
        create_seq_group_metadata_from_prompts(
            prompts,
            num_gpu_blocks,
            block_size,
            continuations=prev_output_tokens,
            final_seq_lens=final_seq_lens,
            num_tokens_processed=num_tokens_processed))

    worker.captured_model = MagicMock()
    worker.execute_model(execute_model_data)

    print(f'{worker.captured_model.execute_if_capturable.call_count=}')

    call_args_list = worker.captured_model.execute_if_capturable.call_args_list
    assert len(call_args_list) == 1
    kwargs = SimpleNamespace(**call_args_list[0].kwargs)

    num_new_tokens = sum(num_missing_from_kv_cache)
    padded_num_new_tokens = math.ceil(num_new_tokens / 8) * 8
    # Expect the number of tokens being saved to KV cache to equal the total
    # number of tokens missing from KV cache.
    assert kwargs.input_metadata.num_valid_tokens == padded_num_new_tokens
    assert kwargs.input_metadata.num_prompt_tokens == 0
    assert kwargs.input_metadata.slot_mapping.shape[0] == padded_num_new_tokens
    assert kwargs.input_metadata.num_generation_tokens == num_new_tokens

    expected_positions = []
    expected_input_ids = []
    for prompt_token_ids, output_tokens, num_tok_missing_from_kv_cache in zip(
            prompts, prev_output_tokens, num_missing_from_kv_cache):
        seq = prompt_token_ids + output_tokens
        total_seq_len = len(seq)
        for i in range(num_tok_missing_from_kv_cache):
            position = total_seq_len - num_tok_missing_from_kv_cache + i
            expected_positions.append(position)
            expected_input_ids.append(seq[position])

    print(f'{expected_positions=}')
    print(f'{expected_input_ids=}')

    print(f'{kwargs.input_ids=}')
    print(f'{kwargs.positions=}')

    # Assert input ids and positions (sans padding) equal to expected.
    assert kwargs.input_ids[:sum(num_missing_from_kv_cache)].tolist(
    ) == expected_input_ids
    assert kwargs.positions[:sum(num_missing_from_kv_cache)].tolist(
    ) == expected_positions


# @pytest.mark.skip("Skip for now")
def test_worker_prepare_inputs_for_prompt():
    seed = 100
    block_size = 16
    num_gpu_blocks = 2048 // block_size
    worker = create_worker(Worker,
                           model_name='JackFram/llama-68m',
                           seed=seed,
                           block_size=block_size,
                           num_gpu_blocks=num_gpu_blocks)
    for batch_size in range(256):
        prompt_lens = []
        seq_group_metadata_list = []
        for i in range(batch_size):
            # make sure all tokens fit into one block
            prompt_len = i % (worker.block_size - 1) + 1
            prompt_lens.append(prompt_len)
            seq_data = list(range(prompt_len))
            seq_group_metadata_list.append(
                SequenceGroupMetadata(
                    request_id=f"test_{i}",
                    is_prompt=True,
                    seq_data={
                        0:
                        SequenceData(seq_data,
                                     prefill_start=0,
                                     prefill_end=prompt_len)
                    },
                    sampling_params=SamplingParams(temperature=0),
                    block_tables={0: [1]},
                    is_chunked_prefill=False,
                    lora_request=None,
                ))
        expected_selected_token_indices = []
        selected_token_start_idx = 0
        for prompt_len in prompt_lens:
            expected_selected_token_indices.append(selected_token_start_idx +
                                                   prompt_len - 1)
            selected_token_start_idx += prompt_len
        input_tokens, input_positions, input_metadata, _, _ = worker._prepare_inputs(
            seq_group_metadata_list)
        assert input_tokens.shape == input_positions.shape == (
            math.ceil(sum(prompt_lens) / 8) * 8, )
        torch.testing.assert_close(input_tokens, input_positions)
        actual = input_metadata.selected_token_indices
        expected = torch.tensor(expected_selected_token_indices,
                                device=actual.device,
                                dtype=actual.dtype)
        torch.testing.assert_close(actual, expected)
