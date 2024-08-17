from array import array
from typing import List

import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import (VLLM_TOKEN_ID_ARRAY_TYPE, SamplingParams,
                           SequenceData, SequenceGroupMetadata)
from vllm.utils import is_cpu
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner

# CUDA graph scenarios to test
#
# Currently CUDA graph is not supported
ENFORCE_EAGER = [True]

BATCH_SIZES = [1, 4, 16, 64, 256]


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
        prompt_adapter_config=engine_config.prompt_adapter_config,
        is_driver_worker=True,
    )
    return model_runner


@pytest.mark.skipif(condition=is_cpu(),
                    reason="CPU backend is currently "
                    "unsupported for encoder/ "
                    "decoder models")
@pytest.mark.parametrize("enforce_eager", ENFORCE_EAGER)
def test_empty_seq_group(enforce_eager, ):
    """Verify prepare prompt and decode returns empty output
       for empty seq group list"""

    model_runner = _create_model_runner(
        "facebook/bart-base",
        seed=0,
        dtype="float16",
        max_num_batched_tokens=100000,
        max_num_seqs=100000,
        enable_chunked_prefill=False,
        enforce_eager=enforce_eager,
    )
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    model_input = model_runner._prepare_model_input_tensors(
        seq_group_metadata_list)
    (
        input_tokens,
        input_positions,
        encoder_input_tokens,
        encoder_input_positions,
        attn_metadata,
        return_seq_lens,
    ) = (
        model_input.input_tokens,
        model_input.input_positions,
        model_input.encoder_input_tokens,
        model_input.encoder_input_positions,
        model_input.attn_metadata,
        model_input.seq_lens,
    )
    assert input_tokens is None
    assert input_positions is None
    assert encoder_input_tokens is None
    assert encoder_input_positions is None
    assert attn_metadata is None
    assert return_seq_lens is None


@pytest.mark.skipif(condition=is_cpu(),
                    reason="CPU backend is currently "
                    "unsupported for encoder/ "
                    "decoder models")
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("enforce_eager", ENFORCE_EAGER)
def test_prepare_prompt(
    batch_size,
    enforce_eager,
):
    '''
    Test the ability of the encoder/decoder model runner subclass to
    produce prefill-phase model inputs & attention metadata.

    Test behavior:

    * Instantiate BART base model & enc/dec model runner
    * Construct sequence-group metadata for dummy prompts
    * Test that encoder attention, decoder self-attention,
      and encoder/decoder cross-attention inputs are correct

    Arguments:

    * batch_size
    * backend_name: The attention backend under test
    * enforce_eager: Enforce eager mode if True (i.e. no CUDAGraph)
    '''

    model_runner = _create_model_runner(
        "facebook/bart-base",
        seed=0,
        dtype="float16",
        max_num_batched_tokens=100000,
        max_num_seqs=100000,
        enable_chunked_prefill=False,
        enforce_eager=enforce_eager,
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
        seq_data = SequenceData(array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                      range(seq_len)))
        encoder_seq_len = (i + 1) % (model_runner.block_size - 1) + 1
        encoder_seq_lens.append(encoder_seq_len)
        encoder_seq_data = SequenceData(
            array(VLLM_TOKEN_ID_ARRAY_TYPE, range(encoder_seq_len)))
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=True,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables=block_tables,
            encoder_seq_data=encoder_seq_data,
            cross_block_table=cross_block_table,
        )
        assert seq_group_metadata.token_chunk_size == seq_data.get_len()
        seq_group_metadata_list.append(seq_group_metadata)

    # Build
    # * Decoder model inputs
    # * Decoder self-attention KV caching data structures
    # * Encoder model inputs
    # * Encoder/decoder cross-attention KV caching data structures
    model_input = model_runner.prepare_model_input(seq_group_metadata_list)

    input_tokens = model_input.input_tokens
    input_positions = model_input.input_positions
    attn_metadata = model_input.attn_metadata
    return_seq_lens = model_input.seq_lens
    slot_mapping = attn_metadata.slot_mapping
    encoder_input_tokens = model_input.encoder_input_tokens
    encoder_input_positions = model_input.encoder_input_positions
    cross_slot_mapping = attn_metadata.cross_slot_mapping
    assert return_seq_lens == seq_lens
    assert len(slot_mapping) == len(input_tokens)
    assert len(cross_slot_mapping) == len(encoder_input_tokens)

    # Verify input metadata is correct for prompts.
    # - Decoder attention metadata
    device = model_runner.device
    assert attn_metadata.num_prefills > 0
    assert attn_metadata.num_decode_tokens == 0
    assert torch.equal(attn_metadata.seq_lens_tensor,
                       torch.tensor(seq_lens, device=device, dtype=torch.int))
    assert attn_metadata.seq_lens == seq_lens
    assert attn_metadata.max_prefill_seq_len == max(seq_lens)
    assert attn_metadata.max_decode_seq_len == 0
    # - Encoder attention metadata
    assert attn_metadata.encoder_seq_lens == encoder_seq_lens
    assert torch.equal(
        attn_metadata.encoder_seq_lens_tensor,
        torch.tensor(encoder_seq_lens, device=device, dtype=torch.int))
    assert attn_metadata.max_encoder_seq_len == max(encoder_seq_lens)
    assert attn_metadata.num_encoder_tokens == sum(encoder_seq_lens)

    # Test decoder subquery start locs.
    start_idx = 0
    start_loc = [start_idx]
    for seq_len in seq_lens:
        start_idx += seq_len
        start_loc.append(start_idx)
    assert torch.equal(
        attn_metadata.query_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device),
    )

    # Test decoder seq start locs & context lengths

    assert torch.equal(
        attn_metadata.seq_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device),
    )
    assert torch.equal(
        attn_metadata.context_lens_tensor,
        torch.zeros(attn_metadata.context_lens_tensor.shape[0],
                    dtype=torch.int,
                    device=device),
    )

    # Verify block tables are correct for prompts
    # - Decoder self-attention
    expected = torch.tensor(
        [[] for _ in range(len(seq_group_metadata_list))],
        dtype=torch.int32,
        device=model_runner.device,
    )
    assert torch.equal(
        attn_metadata.block_tables,
        expected,
    )
    # - Encoder/decoder cross-attention
    assert torch.equal(
        attn_metadata.cross_block_tables,
        expected,
    )

    # Cuda graph should not be used for prefill.
    assert attn_metadata.use_cuda_graph is False

    # Verify the lengths of input tokens & positions
    # - Decoder
    assert len(input_tokens) == sum(seq_lens)
    assert len(input_positions) == sum(seq_lens)
    # -- An indirect check that model_input.input_tokens
    #    and model_input.input_positions are correct -
    #    by design of the test, the input tokens are
    #    equal to the input position values, so if
    #    the model_input data structure has the correct
    #    values then these two should be equal
    assert torch.equal(
        input_tokens,
        input_positions,
    )
    # - Encoder
    assert len(encoder_input_tokens) == sum(encoder_seq_lens)
    # -- An indirect check that model_input.encoder_input_tokens
    #    and model_input.encoder_input_positions are correct -
    #    by design of the test, the input tokens are
    #    equal to the input position values, so if
    #    the model_input data structure has the correct
    #    values then these two should be equal
    assert torch.equal(
        encoder_input_tokens,
        encoder_input_positions,
    )

    # Test that vLLM sampling infrastructure chooses the correct
    # sequence positions at which to sample (i.e. the end of
    # each sequence) in the prefill phase

    expected_selected_token_indices = []
    selected_token_start_idx = 0
    for seq_len in seq_lens:
        # Compute the index offset of the final token in each
        # prompt (recall that the prompts are concatenated)
        expected_selected_token_indices.append(selected_token_start_idx +
                                               seq_len - 1)
        selected_token_start_idx += seq_len

    sampling_metadata = model_input.sampling_metadata
    actual = sampling_metadata.selected_token_indices
    expected = torch.tensor(
        expected_selected_token_indices,
        device=actual.device,
        dtype=actual.dtype,
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(condition=is_cpu(),
                    reason="CPU backend is currently "
                    "unsupported for encoder/ "
                    "decoder models")
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("enforce_eager", ENFORCE_EAGER)
def test_prepare_decode(
    batch_size,
    enforce_eager,
):
    '''
    Test the ability of the encoder/decoder model runner subclass to
    produce decode-phase model inputs & attention metadata.

    Test behavior:

    * Instantiate BART base model & enc/dec model runner
    * Construct sequence-group metadata for dummy prompts
    * Test that encoder attention, decoder self-attention,
      and encoder/decoder cross-attention inputs are correct

    Arguments:

    * batch_size
    * backend_name: The attention backend under test
    * enforce_eager: Enforce eager mode if True (i.e. no CUDAGraph)
    '''

    model_runner = _create_model_runner(
        "facebook/bart-base",
        seed=0,
        dtype="float16",
        max_num_batched_tokens=100000,
        max_num_seqs=100000,
        enable_chunked_prefill=False,
        enforce_eager=enforce_eager,
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
        seq_data = SequenceData(
            array(VLLM_TOKEN_ID_ARRAY_TYPE, (range(seq_len))))
        encoder_seq_len = (i + 1) % (model_runner.block_size - 1) + 1
        encoder_seq_lens.append(encoder_seq_len)
        encoder_seq_data = SequenceData(
            array(VLLM_TOKEN_ID_ARRAY_TYPE, (range(encoder_seq_len))))
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=False,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables=block_tables,
            encoder_seq_data=encoder_seq_data,
            cross_block_table=cross_block_table,
        )
        assert seq_group_metadata.token_chunk_size == 1
        seq_group_metadata_list.append(seq_group_metadata)

    # Build
    # * Decoder model inputs
    # * Decoder self-attention KV caching data structures
    # * Encoder model inputs
    # * Encoder/decoder cross-attention KV caching data structures
    model_input = model_runner.prepare_model_input(seq_group_metadata_list)
    input_tokens = model_input.input_tokens
    input_positions = model_input.input_positions
    attn_metadata = model_input.attn_metadata
    return_seq_lens = model_input.seq_lens
    slot_mapping = attn_metadata.slot_mapping
    encoder_input_tokens = model_input.encoder_input_tokens
    encoder_input_positions = model_input.encoder_input_positions
    cross_slot_mapping = attn_metadata.cross_slot_mapping
    assert return_seq_lens == seq_lens
    assert len(slot_mapping) == len(input_tokens)
    assert len(cross_slot_mapping) == len(encoder_input_tokens)

    # Verify input metadata is correct for decode phase.
    # - Decoder attention metadata
    device = model_runner.device
    assert attn_metadata.num_prefills == 0
    assert attn_metadata.num_decode_tokens > 0
    assert torch.equal(attn_metadata.seq_lens_tensor,
                       torch.tensor(seq_lens, device=device, dtype=torch.int))
    assert attn_metadata.seq_lens == seq_lens
    assert attn_metadata.max_prefill_seq_len == 0
    assert attn_metadata.max_decode_seq_len == max(seq_lens)
    # - Encoder attention metadata
    assert attn_metadata.encoder_seq_lens == encoder_seq_lens
    assert torch.equal(
        attn_metadata.encoder_seq_lens_tensor,
        torch.tensor(encoder_seq_lens, device=device, dtype=torch.int))
    assert attn_metadata.max_encoder_seq_len == max(encoder_seq_lens)
    assert attn_metadata.num_encoder_tokens == sum(encoder_seq_lens)

    # Test decoder subquery start locs.
    start_idx = 0
    start_loc = [start_idx]
    for seq_len in seq_lens:
        start_idx += 1
        start_loc.append(start_idx)
    assert torch.equal(
        attn_metadata.query_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device),
    )

    # Test decoder seq start locs. Note that for normal prefill it is
    # equivalent to query_start_loc.
    start_idx = 0
    seq_start_loc = [start_idx]
    for seq_len in seq_lens:
        start_idx += seq_len
        seq_start_loc.append(start_idx)

    # Test seq_start_loc and context lengths

    assert torch.equal(
        attn_metadata.seq_start_loc,
        torch.tensor(seq_start_loc, dtype=torch.int32, device=device),
    )
    assert torch.equal(
        attn_metadata.context_lens_tensor,
        torch.tensor([seq_len - 1 for seq_len in seq_lens],
                     dtype=torch.int,
                     device=device))

    # Verify block tables are correct for prompts
    # - Decoder self-attention
    expected = torch.tensor(
        [block_tables[0] for _ in range(len(seq_group_metadata_list))],
        dtype=torch.int32,
        device=model_runner.device)
    assert torch.equal(
        attn_metadata.block_tables,
        expected,
    )
    # - Encoder/decoder cross-attention
    expected = torch.tensor(
        [cross_block_table for _ in range(len(seq_group_metadata_list))],
        dtype=torch.int32,
        device=model_runner.device)
    assert torch.equal(
        attn_metadata.cross_block_tables,
        expected,
    )

    # Cuda graph should is currently not supported for encoder/decoer.
    assert attn_metadata.use_cuda_graph is False

    # Verify the lengths of input tokens & positions
    # - Decoder
    assert len(input_tokens) == len(seq_lens)
    assert len(input_positions) == len(seq_lens)
    # -- An indirect check that model_input.input_tokens
    #    and model_input.input_positions are correct -
    #    by design of the test, the input tokens are
    #    equal to the input position values, so if
    #    the model_input data structure has the correct
    #    values then these two should be equal
    assert torch.equal(
        input_tokens,
        input_positions,
    )
    # - Encoder
    assert len(encoder_input_tokens) == 0
    assert len(encoder_input_tokens) == 0
    # -- An indirect check that model_input.encoder_input_tokens
    #    and model_input.encoder_input_positions are correct -
    #    by design of the test, the input tokens are
    #    equal to the input position values, so if
    #    the model_input data structure has the correct
    #    values then these two should be equal
    assert torch.equal(
        encoder_input_tokens,
        encoder_input_positions,
    )

    # Test that vLLM sampling infrastructure chooses the correct
    # sequence positions at which to sample (i.e. the end of
    # each sequence) in the decode phase

    expected_selected_token_indices = []
    selected_token_start_idx = 0
    for seq_len in seq_lens:
        # Compute the index offset of the final token in each
        # sequence's decoded outputs; since a single token is
        # decoded per iteration per sequence, then the length
        # of the decoded tokens for a given sequence is 1 and
        # the final index offset into a given sequence's
        # generated tokens is 0 (i.e. the expected sampling index
        # for a given sequence is just `selected_token_start_idx`)
        expected_selected_token_indices.append(selected_token_start_idx)
        selected_token_start_idx += 1

    sampling_metadata = model_input.sampling_metadata
    actual = sampling_metadata.selected_token_indices
    expected = torch.tensor(
        expected_selected_token_indices,
        device=actual.device,
        dtype=actual.dtype,
    )
    assert torch.equal(actual, expected)
