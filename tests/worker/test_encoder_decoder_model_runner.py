from typing import List

import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner


def _create_model_runner(model: str, *args,
                         **kwargs) -> EncoderDecoderModelRunner:
    engine_args = EngineArgs(model, *args, **kwargs)
    engine_config = engine_args.create_engine_config()
    model_runner = EncoderDecoderModelRunner(
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
        cache_config=engine_config.cache_config,
        load_config=engine_config.load_config,
        lora_config=engine_config.lora_config,
        is_driver_worker=True,
    )
    return model_runner


@pytest.mark.parametrize("batch_size", list(range(1, 257)))
def test_prepare_prompt(batch_size):
    model_runner = _create_model_runner(
        "facebook/bart-base",
        max_num_batched_tokens=100000,
        max_num_seqs=100000,
        enable_chunked_prefill=False,
    )

    seq_lens: List[int] = []
    encoder_seq_lens: List[int] = []
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    block_tables = {0: [1]}
    cross_block_table = [2]
    for i in range(batch_size):
        # make sure all tokens fit into one block
        seq_len = i % (model_runner.block_size - 1) + 1
        seq_lens.append(seq_len)
        seq_data = SequenceData(list(range(seq_len)))
        encoder_seq_len = (i+1) % (model_runner.block_size - 1) + 1
        encoder_seq_lens.append(encoder_seq_len)
        encoder_seq_data = SequenceData(list(range(encoder_seq_len)))
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=True,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables=block_tables,
            encoder_seq_data=encoder_seq_data,
            cross_block_table=cross_block_table
        )
        assert seq_group_metadata.token_chunk_size == seq_data.get_len()
        seq_group_metadata_list.append(seq_group_metadata)

    expected_selected_token_indices = []
    selected_token_start_idx = 0
    for seq_len in seq_lens:
        expected_selected_token_indices.append(selected_token_start_idx +
                                               seq_len - 1)
        selected_token_start_idx += seq_len
    model_input = model_runner._prepare_model_input(seq_group_metadata_list)
    input_tokens = model_input.input_tokens
    input_positions = model_input.input_positions
    attn_metadata = model_input.attn_metadata
    return_seq_lens = model_input.seq_lens
    slot_mapping = model_input.slot_mapping
    assert return_seq_lens == seq_lens
    assert len(slot_mapping) == len(input_tokens)

    # Verify input metadata is correct for prompts.
    device = model_runner.device
    assert attn_metadata.num_prefills > 0
    assert attn_metadata.num_decode_tokens == 0
    assert torch.allclose(
        attn_metadata.seq_lens_tensor,
        torch.tensor(seq_lens, device=device, dtype=torch.int))
    assert attn_metadata.seq_lens == seq_lens
    assert attn_metadata.max_prefill_seq_len == max(seq_lens)
    assert attn_metadata.max_decode_seq_len == 0

    # Test subquery start locs.
    start_idx = 0
    start_loc = [start_idx]
    for seq_len in seq_lens:
        start_idx += seq_len
        start_loc.append(start_idx)
    assert torch.allclose(
        attn_metadata.query_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device))

    # Test seq start locs. Note that for normal prefill it is
    # equivalent to query_start_loc.
    start_idx = 0
    seq_start_loc = [start_idx]
    for seq_len in seq_lens:
        start_idx += seq_len
        seq_start_loc.append(start_idx)

    assert torch.allclose(
        attn_metadata.seq_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device))
    assert torch.allclose(
        attn_metadata.context_lens_tensor,
        torch.zeros(attn_metadata.context_lens_tensor.shape[0],
                    dtype=torch.int,
                    device=device))

    expected = torch.tensor([[] for _ in range(len(seq_group_metadata_list))],
                            dtype=torch.int32,
                            device=model_runner.device)
    assert torch.allclose(attn_metadata.block_tables, expected)
    # Cuda graph should not be used for prerill.
    assert attn_metadata.use_cuda_graph is False

    assert len(input_tokens) == sum(seq_lens)
    assert len(input_positions) == sum(seq_lens)
    torch.testing.assert_close(input_tokens, input_positions)

    sampling_metadata = SamplingMetadata.prepare(
        seq_group_metadata_list,
        seq_lens,
        query_lens=seq_lens,
        device=model_runner.device,
        pin_memory=model_runner.pin_memory)
    assert len(input_tokens) == sum(seq_lens)
    assert len(input_positions) == sum(seq_lens)
    actual = sampling_metadata.selected_token_indices
    expected = torch.tensor(expected_selected_token_indices,
                            device=actual.device,
                            dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)
    torch.allclose(input_tokens, input_positions)

    actual = sampling_metadata.selected_token_indices
    expected = torch.tensor(expected_selected_token_indices,
                            device=actual.device,
                            dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_empty_seq_group():
    """Verify prepare prompt and decode returns empty output."""
    model_runner = _create_model_runner(
        "facebook/bart-base",
        seed=0,
        dtype="float16",
        enforce_eager=False,
    )
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    model_input = model_runner._prepare_model_input(seq_group_metadata_list)
    input_tokens, input_positions, attn_metadata, slot_mapping = (
        model_input.input_tokens,
        model_input.input_positions,
        model_input.attn_metadata,
        model_input.slot_mapping,
    )
    assert len(input_tokens) == 0
    assert len(input_positions) == 0
    assert attn_metadata is None
    assert len(slot_mapping) == 0

    model_input = model_runner._prepare_model_input(seq_group_metadata_list)
    (input_tokens, input_positions, attn_metadata, slot_mapping,
     return_seq_lens) = (
         model_input.input_tokens,
         model_input.input_positions,
         model_input.attn_metadata,
         model_input.slot_mapping,
         model_input.seq_lens,
     )
    assert len(input_tokens) == 0
    assert len(input_positions) == 0
    assert attn_metadata is None
    assert len(slot_mapping) == 0
    assert len(return_seq_lens) == 0
