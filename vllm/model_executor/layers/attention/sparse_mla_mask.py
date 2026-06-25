# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F

from vllm.vllm_flash_attn.cute import utils
from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@cute.jit
def dense_mask_mod(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    dense_mask = aux_tensors[0]
    word_idx = kv_idx[0] >> 5
    bit_idx = kv_idx[0] & 31
    word = utils.scalar_to_ssa(dense_mask[batch[0], q_idx[0], word_idx], cutlass.Int32)
    one = utils.scalar_to_ssa(1, cutlass.Int32)
    bit = (word >> bit_idx) & one
    zero = utils.scalar_to_ssa(0, cutlass.Int32)
    return bit != zero


def dense_mask_to_block_sparse(
    dense_mask: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 128,
) -> BlockSparseTensorsTorch:
    assert tile_n % 32 == 0, f"tile_n must be a multiple of 32, got {tile_n}"
    batch_size = dense_mask.shape[0]
    words_per_tile = tile_n // 32
    num_m_blocks = _ceildiv(max_seqlen_q, tile_m)
    num_n_blocks = _ceildiv(max_seqlen_k, tile_n)

    padded_q = num_m_blocks * tile_m
    padded_words = num_n_blocks * words_per_tile
    dense_mask = F.pad(
        dense_mask,
        (
            0,
            padded_words - dense_mask.shape[2],
            0,
            padded_q - dense_mask.shape[1],
        ),
    )
    dense_mask = dense_mask.reshape(
        batch_size,
        num_m_blocks,
        tile_m,
        num_n_blocks,
        words_per_tile,
    )
    tile_active = (dense_mask != 0).any(dim=4).any(dim=2)

    mask_block_cnt = tile_active.sum(dim=2).unsqueeze(1).to(torch.int32)
    sort_keys = (~tile_active).to(torch.int32)
    mask_block_idx = sort_keys.argsort(dim=2, stable=True).to(torch.int32)
    mask_block_idx = mask_block_idx.unsqueeze(1)
    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_idx = torch.empty_like(mask_block_idx)

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(tile_m, tile_n),
    )
