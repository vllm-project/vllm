# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.mome_attn import (
    MomeAttentionMetadata,
    MomeAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import SlidingWindowMomeSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


def _make_vllm_config(
    *,
    max_model_len: int = 128,
    max_num_seqs: int = 8,
    num_speculative_tokens: int = 0,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
):
    speculative_config = None
    if num_speculative_tokens > 0:
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
        )
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=speculative_config,
        num_speculative_tokens=num_speculative_tokens,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def _make_spec() -> SlidingWindowMomeSpec:
    return SlidingWindowMomeSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=112,
        dtype=torch.float32,
        sliding_window=BLOCK_SIZE,
        component_dims=(16, 32, 64),
    )


def _make_builder(
    *,
    max_model_len: int = 128,
    max_num_seqs: int = 8,
    num_speculative_tokens: int = 0,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> MomeAttentionMetadataBuilder:
    return MomeAttentionMetadataBuilder(
        kv_cache_spec=_make_spec(),
        layer_names=["layer.0"],
        vllm_config=_make_vllm_config(
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            num_speculative_tokens=num_speculative_tokens,
            cudagraph_mode=cudagraph_mode,
        ),
        device=DEVICE,
    )


def _make_common(
    seq_lens: list[int],
    query_lens: list[int],
    is_prefilling: list[bool],
):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    )
    return common.replace(
        is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool),
    )


def _build(builder: MomeAttentionMetadataBuilder, common, **kwargs):
    return builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        **kwargs,
    )


def test_mome_build_decode_only_metadata():
    builder = _make_builder()
    common = _make_common(
        seq_lens=[17, 32, 1],
        query_lens=[1, 1, 1],
        is_prefilling=[False, False, False],
    )

    metadata = _build(builder, common)

    assert metadata.num_decodes == 3
    assert metadata.num_decode_tokens == 3
    assert metadata.num_prefills == 0
    assert metadata.num_prefill_tokens == 0
    assert metadata.state_indices_tensor_p is not None
    assert metadata.state_indices_tensor_p.shape[0] == 0
    torch.testing.assert_close(
        metadata.state_indices_tensor_d,
        common.block_table_tensor,
    )
    torch.testing.assert_close(
        metadata.block_idx_last_computed_token_d,
        torch.tensor([0, 1, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.block_idx_last_scheduled_token_d,
        torch.tensor([1, 1, 0], dtype=torch.int32),
    )
    assert metadata.query_start_loc_d is None
    assert metadata.has_initial_states_p is None


def test_mome_build_prefill_only_metadata():
    builder = _make_builder()
    common = _make_common(
        seq_lens=[8, 20],
        query_lens=[8, 4],
        is_prefilling=[True, True],
    )

    metadata = _build(builder, common)

    assert metadata.num_decodes == 0
    assert metadata.num_prefills == 2
    assert metadata.num_prefill_tokens == 12
    assert metadata.state_indices_tensor_d is not None
    assert metadata.state_indices_tensor_d.shape[0] == 0
    torch.testing.assert_close(
        metadata.state_indices_tensor_p,
        common.block_table_tensor,
    )
    torch.testing.assert_close(
        metadata.query_start_loc_p,
        torch.tensor([0, 8, 12], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.num_computed_tokens_p,
        torch.tensor([0, 16], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.has_initial_states_p,
        torch.tensor([False, True]),
    )
    assert metadata.nums_dict is not None
    assert metadata.batch_ptr is not None
    assert metadata.token_chunk_offset_ptr is not None


def test_mome_build_mixed_decode_and_prefill_metadata():
    builder = _make_builder()
    common = _make_common(
        seq_lens=[17, 32, 20],
        query_lens=[1, 1, 4],
        is_prefilling=[False, False, True],
    )

    metadata = _build(builder, common)

    assert metadata.num_decodes == 2
    assert metadata.num_decode_tokens == 2
    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == 4
    torch.testing.assert_close(
        metadata.state_indices_tensor_d,
        common.block_table_tensor[:2],
    )
    torch.testing.assert_close(
        metadata.state_indices_tensor_p,
        common.block_table_tensor[2:],
    )
    torch.testing.assert_close(
        metadata.query_start_loc_p,
        torch.tensor([0, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.num_computed_tokens_p,
        torch.tensor([16], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.block_idx_first_scheduled_token_p,
        torch.tensor([1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.block_idx_last_computed_token_p,
        torch.tensor([0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.block_idx_last_scheduled_token_p,
        torch.tensor([1], dtype=torch.int32),
    )


def test_mome_treats_single_token_prefill_with_prior_state_as_decode():
    builder = _make_builder()
    common = _make_common(
        seq_lens=[8, 1],
        query_lens=[1, 1],
        is_prefilling=[True, True],
    )

    metadata = _build(builder, common)

    assert metadata.num_decodes == 1
    assert metadata.num_prefills == 1
    torch.testing.assert_close(
        metadata.state_indices_tensor_d,
        common.block_table_tensor[:1],
    )
    torch.testing.assert_close(
        metadata.state_indices_tensor_p,
        common.block_table_tensor[1:],
    )
    torch.testing.assert_close(
        metadata.has_initial_states_p,
        torch.tensor([False]),
    )


def test_mome_spec_decode_updates_decode_metadata():
    builder = _make_builder(num_speculative_tokens=2)
    common = _make_common(
        seq_lens=[19, 7],
        query_lens=[3, 3],
        is_prefilling=[False, False],
    )
    num_accepted_tokens = torch.tensor([2, 1], dtype=torch.int32, device=DEVICE)
    num_prompt_tokens = torch.tensor([10, 7], dtype=torch.int32)

    metadata = _build(
        builder,
        common,
        num_accepted_tokens=num_accepted_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )

    assert metadata.num_decodes == 2
    assert metadata.num_prefills == 0
    torch.testing.assert_close(metadata.query_start_loc_d, common.query_start_loc)
    torch.testing.assert_close(metadata.num_accepted_tokens, num_accepted_tokens)
    assert metadata.max_decode_query_len == 3
    # Req0 is beyond the prompt and uses the speculative previous-schedule
    # adjustment. Req1 is still at the prompt boundary and keeps the base index.
    torch.testing.assert_close(
        metadata.block_idx_last_computed_token_d,
        torch.tensor([1, 0], dtype=torch.int32),
    )


def test_mome_cudagraph_capture_pads_decode_metadata():
    builder = _make_builder(
        max_model_len=64,
        max_num_seqs=4,
        num_speculative_tokens=2,
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    state_indices = torch.tensor([[11, 12], [21, 22]], dtype=torch.int32, device=DEVICE)
    metadata = MomeAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=6,
        num_reqs=4,
        has_initial_states_p=None,
        query_start_loc_p=None,
        num_computed_tokens_p=None,
        state_indices_tensor_p=torch.empty((0, 2), dtype=torch.int32, device=DEVICE),
        state_indices_tensor_d=state_indices,
        query_start_loc_d=torch.tensor([0, 3, 6, 6, 6], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([2, 1], dtype=torch.int32),
        block_idx_first_scheduled_token_p=None,
        block_idx_last_scheduled_token_p=None,
        block_idx_last_computed_token_p=None,
        block_idx_last_scheduled_token_d=torch.tensor([3, 4], dtype=torch.int32),
        block_idx_last_computed_token_d=torch.tensor([2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([20, 30, 0, 0], dtype=torch.int32),
    )

    padded = builder._update_metadata_for_cudagraph_capture(metadata)

    assert padded.state_indices_tensor_d is not None
    assert padded.state_indices_tensor_d.shape == (4, 4)
    torch.testing.assert_close(padded.state_indices_tensor_d[:2, :2], state_indices)
    assert torch.all(padded.state_indices_tensor_d[2:] == NULL_BLOCK_ID)
    torch.testing.assert_close(
        padded.block_idx_last_scheduled_token_d,
        torch.tensor([3, 4, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        padded.block_idx_last_computed_token_d,
        torch.tensor([2, 3, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        padded.num_accepted_tokens,
        torch.tensor([2, 1, 1, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        padded.query_start_loc_d,
        torch.tensor([0, 3, 6, 6, 6], dtype=torch.int32),
    )


def test_mome_update_block_table_splits_decode_and_prefill_rows():
    builder = _make_builder()
    common = _make_common(
        seq_lens=[17, 32, 20],
        query_lens=[1, 1, 4],
        is_prefilling=[False, False, True],
    )
    metadata = _build(builder, common)
    new_block_table = torch.tensor(
        [[101, 102], [201, 202], [301, 302]],
        dtype=torch.int32,
        device=DEVICE,
    )

    updated = builder.update_block_table(
        metadata,
        new_block_table,
        slot_mapping=torch.empty(0, dtype=torch.int64, device=DEVICE),
    )

    torch.testing.assert_close(updated.state_indices_tensor_d, new_block_table[:2])
    torch.testing.assert_close(updated.state_indices_tensor_p, new_block_table[2:])


def test_mome_update_block_table_rejects_wrong_request_count():
    builder = _make_builder()
    common = _make_common(
        seq_lens=[17, 32, 20],
        query_lens=[1, 1, 4],
        is_prefilling=[False, False, True],
    )
    metadata = _build(builder, common)

    with pytest.raises(AssertionError, match="Mismatch in number of requests"):
        builder.update_block_table(
            metadata,
            torch.zeros((2, 2), dtype=torch.int32, device=DEVICE),
            slot_mapping=torch.empty(0, dtype=torch.int64, device=DEVICE),
        )
