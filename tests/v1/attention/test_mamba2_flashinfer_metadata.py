# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch

from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadata,
    Mamba2AttentionMetadataBuilder,
    compute_flashinfer_ssd_metadata,
    map_flashinfer_state_cache_rows,
)


def test_flashinfer_ssd_metadata_splits_all_boundary_types():
    # Sequence 0 crosses a physical 128-token boundary. Sequence 1 begins at
    # packed offset 200 with an existing 1100-token state, so its 1152-token
    # cache checkpoint lies at packed offset 252, before the next physical
    # boundary at 256.
    metadata = compute_flashinfer_ssd_metadata(
        torch.tensor([0, 200, 400], dtype=torch.int32),
        torch.tensor([0, 1100], dtype=torch.int32),
        chunk_size=128,
        mamba_block_size=1152,
    )

    assert metadata.valid_seqlen == 400
    assert metadata.padded_seqlen == 512
    assert metadata.chunk_indices == [0, 1, 1, 1, 2, 2, 3]
    assert metadata.chunk_offsets == [0, 0, 72, 124, 0, 124, 0]
    assert metadata.seq_chunk_cumsum == [0, 2, 7]
    assert metadata.seq_idx[:200] == [0] * 200
    assert metadata.seq_idx[200:400] == [1] * 200
    # Padded tokens carry a valid id but are masked by valid_seqlen.
    assert metadata.seq_idx[400:] == [1] * 112


@pytest.mark.parametrize("total", [32343, 32344, 32346, 32768])
def test_flashinfer_ssd_metadata_supports_target_packed_lengths(total: int):
    metadata = compute_flashinfer_ssd_metadata(
        torch.tensor([0, total], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        chunk_size=128,
        mamba_block_size=1152,
    )

    expected_padded = ((total + 127) // 128) * 128
    assert metadata.valid_seqlen == total
    assert metadata.padded_seqlen == expected_padded
    assert len(metadata.seq_idx) == expected_padded
    assert metadata.seq_chunk_cumsum == [0, len(metadata.chunk_indices)]
    assert all(
        0 <= offset < 128 for offset in metadata.chunk_offsets
    )
    assert all(
        later > earlier
        for earlier, later in zip(
            metadata.chunk_indices, metadata.chunk_indices[1:]
        )
    )

def test_flashinfer_ssd_metadata_preserves_sequence_relative_chunk_boundaries():
    # Triton first finishes the original sequence chunk containing context token
    # 0, then advances in 128-token chunks: query endpoints 127, 255, 300.
    # Packed-physical FI boundaries alone would instead be 128, 256, 300.
    metadata = compute_flashinfer_ssd_metadata(
        torch.tensor([0, 300], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        chunk_size=128,
        mamba_block_size=1152,
    )

    starts = [
        chunk * 128 + offset
        for chunk, offset in zip(metadata.chunk_indices, metadata.chunk_offsets)
    ]
    ends = starts[1:] + [metadata.valid_seqlen]

    # Retain both contracts: every packed-physical boundary needed by FI and
    # every sequence-relative boundary used by vLLM's Triton recurrence.
    assert starts == [0, 127, 128, 255, 256]
    assert ends == [127, 128, 255, 256, 300]
    assert {127, 255, 300}.issubset(ends)
    assert {128, 256}.issubset(ends)


def test_flashinfer_ssd_output_metadata_uses_only_physical_chunk_boundaries():
    metadata = compute_flashinfer_ssd_metadata(
        torch.tensor([0, 300], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        chunk_size=128,
        mamba_block_size=1152,
        require_triton_state_boundaries=False,
    )

    starts = [
        chunk * 128 + offset
        for chunk, offset in zip(metadata.chunk_indices, metadata.chunk_offsets)
    ]
    assert starts == [0, 128, 256]
    assert metadata.chunk_offsets == [0, 0, 0]


def test_flashinfer_ssd_repack_matches_sequence_relative_chunk_grid():
    metadata = compute_flashinfer_ssd_metadata(
        torch.tensor([0, 300], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        chunk_size=128,
        mamba_block_size=1152,
        require_triton_state_boundaries=False,
        repack_sequence_chunks=True,
    )

    # Triton processes this continuation as 127, 128, and 45 real tokens.
    # Each ragged chunk gets a dedicated zero-padded FI block, so source token
    # 127 skips destination slot 127 and begins the second block.
    assert metadata.chunk_indices == [0, 1, 2]
    assert metadata.chunk_offsets == [0, 0, 0]
    assert metadata.seq_chunk_cumsum == [0, 3]
    assert metadata.valid_seqlen == metadata.padded_seqlen == 384
    assert metadata.token_dst_indices[:127] == list(range(127))
    assert metadata.token_dst_indices[127] == 128
    assert metadata.token_dst_indices[-1] == 300
    assert metadata.segment_seq_ids == [0, 0, 0]
    assert metadata.segment_state_block_indices == [-1, -1, 0]
    assert metadata.requires_repacking


def test_flashinfer_ssd_repack_isolates_mixed_sequence_chunks():
    metadata = compute_flashinfer_ssd_metadata(
        torch.tensor([0, 257, 8192], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        chunk_size=128,
        mamba_block_size=1152,
        require_triton_state_boundaries=False,
        repack_sequence_chunks=True,
    )

    # 257 tokens require three blocks; the second request begins at block 3
    # instead of inheriting packed-buffer phase 1 from source offset 257.
    assert metadata.seq_chunk_cumsum == [0, 3, 65]
    assert metadata.chunk_indices == list(range(65))
    assert metadata.chunk_offsets == [0] * 65
    assert metadata.token_dst_indices[257] == 3 * 128
    assert metadata.valid_seqlen == metadata.padded_seqlen == 65 * 128
    assert metadata.requires_repacking


def test_flashinfer_ssd_repack_selects_checkpoints_and_final_state():
    metadata = compute_flashinfer_ssd_metadata(
        torch.tensor([0, 1400], dtype=torch.int32),
        torch.tensor([1100], dtype=torch.int32),
        chunk_size=128,
        mamba_block_size=1152,
        require_triton_state_boundaries=False,
        repack_sequence_chunks=True,
    )

    # Original-sequence endpoints are 1152, 1280, ..., 2432, 2500.
    # Completed cache blocks 0 and 1 plus final partial block 2 are stored.
    assert len(metadata.chunk_indices) == 12
    assert metadata.segment_seq_ids == [0] * 12
    assert metadata.segment_state_block_indices == [
        0,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        2,
    ]


def test_flashinfer_state_cache_rows_follow_current_block_table():
    block_table = torch.tensor(
        [[31, 32, 33], [41, 42, 43]], dtype=torch.int32
    )
    seq_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    block_indices = torch.tensor([0, -1, 1, 2], dtype=torch.int64)

    actual = map_flashinfer_state_cache_rows(
        block_table, seq_ids, block_indices
    )
    assert actual.dtype == torch.int32
    assert actual.tolist() == [31, -1, 42, 43]


def test_flashinfer_update_block_table_remaps_direct_state_rows(monkeypatch):
    metadata = Mamba2AttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=4,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=2,
        has_initial_states_p=None,
        query_start_loc_p=None,
        num_computed_tokens_p=None,
        state_indices_tensor_p=torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.int32
        ),
        state_indices_tensor_d=None,
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=torch.tensor([128, 128], dtype=torch.int32),
        fi_segment_seq_ids_p=torch.tensor([0, 0, 1], dtype=torch.int64),
        fi_segment_state_block_indices_p=torch.tensor(
            [0, -1, 1], dtype=torch.int64
        ),
        fi_intermediate_state_indices_p=torch.tensor(
            [1, -1, 4], dtype=torch.int32
        ),
    )
    new_block_table = torch.tensor(
        [[31, 32], [41, 42]], dtype=torch.int32
    )

    def fake_base_update(self, metadata, blk_table, slot_mapping):
        return replace(metadata, state_indices_tensor_p=blk_table)

    monkeypatch.setattr(
        BaseMambaAttentionMetadataBuilder,
        "update_block_table",
        fake_base_update,
    )
    builder = object.__new__(Mamba2AttentionMetadataBuilder)
    remapped = builder.update_block_table(
        metadata,
        new_block_table,
        torch.empty(0, dtype=torch.int64),
    )

    assert remapped.fi_intermediate_state_indices_p.tolist() == [31, -1, 42]


@pytest.mark.parametrize(
    "query_start_loc,contexts,error",
    [
        ([1, 2], [0], "start at zero"),
        ([0, 2, 1], [0, 0], "nonnegative"),
        ([0, 1], [0, 0], "one more entry"),
    ],
)
def test_flashinfer_ssd_metadata_rejects_invalid_inputs(
    query_start_loc, contexts, error
):
    with pytest.raises(ValueError, match=error):
        compute_flashinfer_ssd_metadata(
            torch.tensor(query_start_loc),
            torch.tensor(contexts),
            chunk_size=128,
            mamba_block_size=1152,
        )
