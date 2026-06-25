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
