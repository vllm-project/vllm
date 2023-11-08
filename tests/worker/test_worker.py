# pylint: disable=protected-access
import random
import torch

from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.worker.worker import Worker


def test_worker_prepare_inputs_for_prompt():
    worker = Worker(None, None, None)
    worker.block_size = 16
    batch_size = random.randint(1, 256)
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
                seq_data={0: SequenceData(seq_data)},
                sampling_params=SamplingParams(temperature=0),
                block_tables={0: [1]},
            ))
    expected_selected_token_indices = []
    selected_token_start_idx = 0
    max_seq_len = max(prompt_lens)
    for prompt_len in prompt_lens:
        expected_selected_token_indices.append(selected_token_start_idx +
                                               prompt_len - 1)
        selected_token_start_idx += max_seq_len
    input_tokens, input_positions, input_metadata = worker._prepare_inputs(
        seq_group_metadata_list)
    assert input_tokens.shape == input_positions.shape == (batch_size,
                                                           max_seq_len)
    torch.testing.assert_close(input_tokens, input_positions)
    actual = input_metadata.selected_token_indices
    expected = torch.tensor(expected_selected_token_indices,
                            device=actual.device,
                            dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)
