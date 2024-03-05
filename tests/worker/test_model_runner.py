import random
import torch

from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.worker.model_runner import ModelRunner


def test_prepare_prompt():
    model_runner = ModelRunner(None, None, None, None, None)
    model_runner.set_block_size(16)

    batch_size = random.randint(1, 256)
    prompt_lens = []
    seq_group_metadata_list = []
    for i in range(batch_size):
        # make sure all tokens fit into one block
        prompt_len = i % (model_runner.block_size - 1) + 1
        prompt_lens.append(prompt_len)
        seq_data = list(range(prompt_len))
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                is_chunked_prefill=False,
                seq_data={0: SequenceData(seq_data)},
                sampling_params=SamplingParams(temperature=0),
                block_tables={0: [1]},
            ))

    # Advanced prefill range.
    for seq_group_meta, prompt_len in zip(seq_group_metadata_list,
                                          prompt_lens):
        for _, data in seq_group_meta.seq_data.items():
            data.advance_prefill_range(prompt_len)

    expected_selected_token_indices = []
    selected_token_start_idx = 0
    max_seq_len = max(prompt_lens)
    for prompt_len in prompt_lens:
        expected_selected_token_indices.append(selected_token_start_idx +
                                               prompt_len - 1)
        selected_token_start_idx += max_seq_len
    input_tokens, input_positions, _, return_prompt_lens, _, _, _, _ = (
        model_runner._prepare_prompt(seq_group_metadata_list))
    assert return_prompt_lens == prompt_lens
    sampling_metadata = model_runner._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens=prompt_lens)

    def round_up_to_next_multiple_of_8(n):
        return ((n + 7) // 8) * 8

    assert input_tokens.shape == (round_up_to_next_multiple_of_8(
        sum(prompt_lens)), )
    assert input_positions.shape == (round_up_to_next_multiple_of_8(
        sum(prompt_lens)), )
    torch.testing.assert_close(input_tokens, input_positions)

    actual = sampling_metadata.selected_token_indices
    expected = torch.tensor(expected_selected_token_indices,
                            device=actual.device,
                            dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)


# TODO(sang) Test chunked prefill prompt.
