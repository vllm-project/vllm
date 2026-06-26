# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.attention.sparse_mla_mask import (
    dense_mask_to_block_sparse,
)


def _pack_dense_mask(dense_mask: torch.Tensor) -> torch.Tensor:
    padded_k = (32 - dense_mask.shape[-1] % 32) % 32
    if padded_k:
        dense_mask = torch.nn.functional.pad(dense_mask, (0, padded_k))
    shifts = torch.arange(32, dtype=torch.int32)
    return (
        dense_mask.reshape(*dense_mask.shape[:-1], -1, 32).to(torch.int32) << shifts
    ).sum(dim=-1, dtype=torch.int32)


def test_dense_mask_to_block_sparse_matches_dense_tiles() -> None:
    dense_mask = torch.zeros(2, 5, 130, dtype=torch.bool)
    dense_mask[0, 0, 3] = True
    dense_mask[0, 4, 129] = True
    dense_mask[1, 2, 64] = True
    dense_mask[1, 3, 65] = True

    sparse = dense_mask_to_block_sparse(
        _pack_dense_mask(dense_mask),
        max_seqlen_q=5,
        max_seqlen_k=130,
        tile_m=4,
        tile_n=64,
    )

    assert sparse.block_size == (4, 64)
    torch.testing.assert_close(
        sparse.mask_block_cnt,
        torch.tensor([[[1, 1]], [[1, 0]]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.mask_block_idx[:, :, :, :2],
        torch.tensor(
            [
                [[[0, 1], [2, 0]]],
                [[[1, 0], [0, 1]]],
            ],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        sparse.full_block_cnt,
        torch.zeros_like(sparse.mask_block_cnt),
    )
    assert sparse.full_block_idx is not None
    assert sparse.full_block_idx.shape == sparse.mask_block_idx.shape


def test_dense_mask_to_block_sparse_varlen_uses_packed_layout() -> None:
    dense_mask = torch.zeros(2, 5, 130, dtype=torch.bool)
    dense_mask[0, 0, 3] = True
    dense_mask[0, 4, 129] = True
    dense_mask[1, 2, 63] = True

    sparse = dense_mask_to_block_sparse(
        _pack_dense_mask(dense_mask),
        max_seqlen_q=5,
        max_seqlen_k=130,
        seq_lens_q=[5, 3],
        seq_lens_k=[130, 64],
        tile_m=4,
        tile_n=64,
    )

    torch.testing.assert_close(
        sparse.mask_block_cnt,
        torch.tensor([[1, 1, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.mask_block_idx,
        torch.tensor([[0, 1, 2, 2, 0, 1, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.cu_total_m_blocks,
        torch.tensor([0, 2, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.cu_block_idx_offsets,
        torch.tensor([0, 6, 7], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.full_block_cnt,
        torch.zeros_like(sparse.mask_block_cnt),
    )
    assert sparse.full_block_idx is not None
    assert sparse.full_block_idx.shape == sparse.mask_block_idx.shape


def test_dense_mask_to_block_sparse_splits_full_and_partial_blocks() -> None:
    dense_mask = torch.zeros(1, 4, 64, dtype=torch.bool)
    dense_mask[0, :2, :32] = True
    dense_mask[0, 2, 40] = True

    sparse = dense_mask_to_block_sparse(
        _pack_dense_mask(dense_mask),
        max_seqlen_q=4,
        max_seqlen_k=64,
        seq_lens_q=[4],
        seq_lens_k=[64],
        tile_m=2,
        tile_n=32,
    )

    torch.testing.assert_close(
        sparse.mask_block_cnt,
        torch.tensor([[0, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.mask_block_idx,
        torch.tensor([[0, 1, 1, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.full_block_cnt,
        torch.tensor([[1, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.full_block_idx,
        torch.tensor([[0, 1, 0, 1]], dtype=torch.int32),
    )


def test_dense_mask_to_block_sparse_varlen_equal_blocks_vectorized() -> None:
    dense_mask = torch.zeros(2, 4, 64, dtype=torch.bool)
    dense_mask[0, :2, :32] = True
    dense_mask[0, 2, 40] = True
    dense_mask[1, 0, 5] = True
    dense_mask[1, 2:, 32:] = True

    sparse = dense_mask_to_block_sparse(
        _pack_dense_mask(dense_mask),
        max_seqlen_q=4,
        max_seqlen_k=64,
        seq_lens_q=[4, 4],
        seq_lens_k=[64, 64],
        tile_m=2,
        tile_n=32,
    )

    torch.testing.assert_close(
        sparse.mask_block_cnt,
        torch.tensor([[0, 1, 1, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.mask_block_idx,
        torch.tensor([[0, 1, 1, 0, 0, 1, 0, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.full_block_cnt,
        torch.tensor([[1, 0, 0, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.full_block_idx,
        torch.tensor([[0, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.cu_total_m_blocks,
        torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sparse.cu_block_idx_offsets,
        torch.tensor([0, 4, 8], dtype=torch.int32),
    )
