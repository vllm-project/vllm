from typing import List

import pytest
import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.utils import get_open_port
from vllm.worker.model_runner import ModelRunner


def _create_model_runner(model: str, *args, **kwargs) -> ModelRunner:
    engine_args = EngineArgs(model, *args, **kwargs)
    engine_config = engine_args.create_engine_config()
    model_runner = ModelRunner(
        vllm_config=engine_config,
        is_driver_worker=True,
    )
    return model_runner


@pytest.mark.parametrize("batch_size", list(range(1, 257)))
def test_prepare_prompt(batch_size):
    model_runner = _create_model_runner(
        "facebook/opt-125m",
        max_num_batched_tokens=100000,
        max_num_seqs=100000,
        enable_chunked_prefill=False,
    )

    seq_lens: List[int] = []
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    block_tables = {0: [1]}
    for i in range(batch_size):
        # make sure all tokens fit into one block
        seq_len = i % (model_runner.block_size - 1) + 1
        seq_lens.append(seq_len)
        seq_data = SequenceData.from_seqs(range(seq_len))
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=True,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables=block_tables,
        )
        assert seq_group_metadata.token_chunk_size == seq_data.get_len()
        seq_group_metadata_list.append(seq_group_metadata)

    expected_selected_token_indices = []
    selected_token_start_idx = 0
    for seq_len in seq_lens:
        expected_selected_token_indices.append(selected_token_start_idx +
                                               seq_len - 1)
        selected_token_start_idx += seq_len
    model_input = model_runner._prepare_model_input_tensors(
        seq_group_metadata_list)
    input_tokens = model_input.input_tokens
    input_positions = model_input.input_positions
    attn_metadata = model_input.attn_metadata
    return_seq_lens = model_input.seq_lens
    slot_mapping = attn_metadata.slot_mapping
    assert return_seq_lens == seq_lens
    assert len(slot_mapping) == len(input_tokens)

    # Verify input metadata is correct for prompts.
    device = model_runner.device
    assert attn_metadata.num_prefills > 0
    assert attn_metadata.num_decode_tokens == 0
    torch.testing.assert_close(
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
    torch.testing.assert_close(
        attn_metadata.query_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device))

    # Test seq start locs. Note that for normal prefill it is
    # equivalent to query_start_loc.
    start_idx = 0
    seq_start_loc = [start_idx]
    for seq_len in seq_lens:
        start_idx += seq_len
        seq_start_loc.append(start_idx)

    torch.testing.assert_close(
        attn_metadata.seq_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device))
    torch.testing.assert_close(
        attn_metadata.context_lens_tensor,
        torch.zeros(attn_metadata.context_lens_tensor.shape[0],
                    dtype=torch.int,
                    device=device))

    expected = torch.tensor([[] for _ in range(len(seq_group_metadata_list))],
                            dtype=torch.int32,
                            device=model_runner.device)
    torch.testing.assert_close(attn_metadata.block_tables, expected)
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


@pytest.mark.parametrize("batch_size", list(range(1, 257)))
def test_prepare_decode_cuda_graph(batch_size):
    model_runner = _create_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        enforce_eager=False,
        max_num_batched_tokens=100000,
        max_num_seqs=100000,
        enable_chunked_prefill=False,
    )

    context_lens: List[int] = []
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    # Assume each seq group finishes prefill.
    for i in range(batch_size):
        # make sure all tokens fit into one block
        context_len = i % (model_runner.block_size - 1) + 1
        context_lens.append(context_len)
        seq_data = SequenceData.from_seqs(range(context_len))
        seq_data.update_num_computed_tokens(context_len)
        # Append one token ID since prefill is finished.
        seq_data.append_token_id(1, 0)
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=False,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables={0: [1]},
        )
        assert seq_group_metadata.token_chunk_size == 1
        seq_group_metadata_list.append(seq_group_metadata)

    model_input = model_runner._prepare_model_input_tensors(
        seq_group_metadata_list)
    input_tokens, input_positions, attn_metadata, slot_mapping = (
        model_input.input_tokens, model_input.input_positions,
        model_input.attn_metadata, model_input.attn_metadata.slot_mapping)
    assert len(slot_mapping) == len(input_tokens)

    expected_bs = VllmConfig.get_graph_batch_size(len(seq_group_metadata_list))
    # Verify input metadata is correct for prompts.
    device = model_runner.device
    assert attn_metadata.num_prefills == 0
    assert attn_metadata.num_prefill_tokens == 0
    seq_lens = [context_len + 1 for context_len in context_lens]
    # seq_lens are padded to expected_bs
    for _ in range(expected_bs - len(seq_lens)):
        seq_lens.append(1)
    assert attn_metadata.seq_lens == seq_lens
    assert attn_metadata.num_decode_tokens == len(seq_lens)
    start_idx = 0
    start_loc = [start_idx]
    for _ in context_lens:
        # decode has only 1 token for query.
        start_idx += 1
        start_loc.append(start_idx)
    torch.testing.assert_close(
        attn_metadata.query_start_loc,
        torch.tensor(start_loc, dtype=torch.int32, device=device))

    start_idx = 0
    seq_start_loc = [start_idx]
    for seq_len in seq_lens:
        start_idx += seq_len
        seq_start_loc.append(start_idx)
    torch.testing.assert_close(
        attn_metadata.seq_start_loc,
        torch.tensor(seq_start_loc, dtype=torch.int32, device=device))

    torch.testing.assert_close(
        attn_metadata.context_lens_tensor,
        torch.tensor(context_lens, dtype=torch.int, device=device))
    assert attn_metadata.max_decode_seq_len == max(seq_lens)
    torch.testing.assert_close(
        attn_metadata.seq_lens_tensor[:len(seq_lens)],
        torch.tensor(seq_lens, dtype=torch.int, device=device))

    # block table's first index corresponds to each batch, meaning in
    # decoding it is each token.
    assert attn_metadata.block_tables.shape[0] == len(input_tokens)
    # Block table's second dim correspondsd to each token's block number.
    # It is padded up to
    assert attn_metadata.block_tables.shape[1] == (
        model_runner.get_max_block_per_batch())
    assert attn_metadata.use_cuda_graph is True

    assert len(input_tokens) == expected_bs
    assert len(input_positions) == expected_bs
    torch.allclose(input_tokens, input_positions)

    # Verify Sampling
    expected_selected_token_indices = []
    for selected_token_start_idx, _ in enumerate(context_lens):
        expected_selected_token_indices.append(selected_token_start_idx)
    sampling_metadata = SamplingMetadata.prepare(
        seq_group_metadata_list,
        seq_lens,
        # query lens is all 1 for decode.
        query_lens=[1 for _ in range(len(context_lens))],
        device=model_runner.device,
        pin_memory=model_runner.pin_memory)
    actual = sampling_metadata.selected_token_indices
    expected = torch.tensor(expected_selected_token_indices,
                            device=actual.device,
                            dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_empty_seq_group():
    """Verify prepare prompt and decode returns empty output."""
    model_runner = _create_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        enforce_eager=False,
    )
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    model_input = model_runner._prepare_model_input_tensors(
        seq_group_metadata_list)
    input_tokens, input_positions, attn_metadata = (
        model_input.input_tokens,
        model_input.input_positions,
        model_input.attn_metadata,
    )
    assert input_tokens is None
    assert input_positions is None
    assert attn_metadata is None

    model_input = model_runner._prepare_model_input_tensors(
        seq_group_metadata_list)
    (input_tokens, input_positions, attn_metadata, return_seq_lens) = (
        model_input.input_tokens,
        model_input.input_positions,
        model_input.attn_metadata,
        model_input.seq_lens,
    )
    assert input_tokens is None
    assert input_positions is None
    assert attn_metadata is None
    assert return_seq_lens is None


@pytest.fixture
def distributed_init():
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        local_rank=0)
    ensure_model_parallel_initialized(1, 1)


@pytest.mark.parametrize("batch_size", list(range(2, 128)))
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_hybrid_batches(batch_size, enforce_eager, distributed_init):
    model_runner = _create_model_runner(
        "facebook/opt-125m",
        seed=0,
        dtype="float16",
        enforce_eager=enforce_eager,
        max_num_batched_tokens=100000,
        max_num_seqs=100000,
        enable_chunked_prefill=True,
    )

    # Add prefill requests.
    seq_lens: List[int] = []
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    prefill_metadata_list: List[SequenceGroupMetadata] = []
    decode_metadata_list: List[SequenceGroupMetadata] = []
    block_tables = {0: [1]}
    prefill_batch_size = batch_size // 2
    decode_batch_size = batch_size - prefill_batch_size
    for i in range(prefill_batch_size):
        # make sure all tokens fit into one block
        seq_len = i % (model_runner.block_size - 1) + 1
        seq_lens.append(seq_len)
        seq_data = SequenceData.from_seqs(range(seq_len))
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=True,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables=block_tables,
        )
        assert seq_group_metadata.token_chunk_size == seq_data.get_len()
        seq_group_metadata_list.append(seq_group_metadata)
        prefill_metadata_list.append(seq_group_metadata)

    # Add decode requests
    for i in range(prefill_batch_size, batch_size):
        # make sure all tokens fit into one block
        context_len = i % (model_runner.block_size - 1) + 1
        seq_data = SequenceData.from_seqs(range(context_len))
        seq_data.append_token_id(1, 0)
        seq_data.update_num_computed_tokens(context_len)
        seq_group_metadata = SequenceGroupMetadata(
            request_id=f"test_{i}",
            is_prompt=False,
            seq_data={0: seq_data},
            sampling_params=SamplingParams(temperature=0),
            block_tables={0: [1]},
        )
        assert seq_group_metadata.token_chunk_size == 1
        seq_group_metadata_list.append(seq_group_metadata)
        decode_metadata_list.append(seq_group_metadata)

    model_input = model_runner.prepare_model_input(seq_group_metadata_list)
    (input_tokens, input_positions, attn_metadata) = (
        model_input.input_tokens,
        model_input.input_positions,
        model_input.attn_metadata,
    )

    prefill_meta_actual = attn_metadata.prefill_metadata
    decode_meta_actual = attn_metadata.decode_metadata

    assert len(attn_metadata.slot_mapping) == len(input_tokens)
    assert len(input_positions) == len(input_tokens)
    assert attn_metadata.num_prefills == prefill_batch_size
    assert attn_metadata.num_decode_tokens == decode_batch_size
    assert attn_metadata.num_prefill_tokens == sum(seq_lens)

    # Verify attn metadata is consistent. We don't need to test individual
    # values here because they are tested above.
    attn_metadata = model_runner._prepare_model_input_tensors(
        seq_group_metadata_list).attn_metadata

    for attr_expected, attr_actual in zip(vars(attn_metadata.prefill_metadata),
                                          vars(prefill_meta_actual)):
        assert attr_expected[1] == attr_actual[1]
    for attr_expected, attr_actual in zip(vars(attn_metadata.decode_metadata),
                                          vars(decode_meta_actual)):
        assert attr_expected[1] == attr_actual[1]
