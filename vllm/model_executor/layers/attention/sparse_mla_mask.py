# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F

from vllm.vllm_flash_attn.cute import utils
from vllm.vllm_flash_attn.cute.block_sparsity import BlockSparseTensorsTorch


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tiles_to_varlen_block_sparse(
    tile_partial: torch.Tensor,
    tile_full: torch.Tensor,
    seq_lens_q: Sequence[int],
    seq_lens_k: Sequence[int],
    tile_m: int,
    tile_n: int,
) -> BlockSparseTensorsTorch:
    batch_size = tile_partial.shape[0]
    assert len(seq_lens_q) == batch_size
    assert len(seq_lens_k) == batch_size

    num_m_blocks_per_seq = [_ceildiv(seq_len, tile_m) for seq_len in seq_lens_q]
    num_n_blocks_per_seq = [_ceildiv(seq_len, tile_n) for seq_len in seq_lens_k]
    if len(set(num_m_blocks_per_seq)) == 1 and len(set(num_n_blocks_per_seq)) == 1:
        # Varlen FA expects packed 2D metadata, but equal block counts can
        # be packed by flattening the rectangular batch-major layout.
        num_m_blocks_b = num_m_blocks_per_seq[0]
        num_n_blocks_b = num_n_blocks_per_seq[0]
        partial = tile_partial[:, :num_m_blocks_b, :num_n_blocks_b]
        full = tile_full[:, :num_m_blocks_b, :num_n_blocks_b]

        mask_block_cnt = partial.sum(dim=2).reshape(1, -1).to(torch.int32)
        sort_keys = (~partial).to(torch.int32)
        mask_block_idx = (
            sort_keys.argsort(dim=2, stable=True).reshape(1, -1).to(torch.int32)
        )
        full_block_cnt = full.sum(dim=2).reshape(1, -1).to(torch.int32)
        sort_keys = (~full).to(torch.int32)
        full_block_idx = (
            sort_keys.argsort(dim=2, stable=True).reshape(1, -1).to(torch.int32)
        )
        batch_offsets = torch.arange(
            batch_size + 1, dtype=torch.int32, device=tile_partial.device
        )

        return BlockSparseTensorsTorch(
            mask_block_cnt=mask_block_cnt,
            mask_block_idx=mask_block_idx,
            full_block_cnt=full_block_cnt,
            full_block_idx=full_block_idx,
            cu_total_m_blocks=batch_offsets * num_m_blocks_b,
            cu_block_idx_offsets=batch_offsets * num_m_blocks_b * num_n_blocks_b,
            block_size=(tile_m, tile_n),
        )

    cu_total_m_blocks = [0]
    cu_block_idx_offsets = [0]
    total_m_blocks = sum(num_m_blocks_per_seq)
    total_block_idx = sum(
        num_m_blocks * num_n_blocks
        for num_m_blocks, num_n_blocks in zip(
            num_m_blocks_per_seq, num_n_blocks_per_seq
        )
    )
    mask_block_cnt = torch.empty(
        (1, total_m_blocks), dtype=torch.int32, device=tile_partial.device
    )
    mask_block_idx = torch.empty(
        (1, total_block_idx), dtype=torch.int32, device=tile_partial.device
    )
    full_block_cnt = torch.empty_like(mask_block_cnt)
    full_block_idx = torch.empty_like(mask_block_idx)

    # Ragged varlen metadata has different rows widths per request, so each
    # request must be compacted before concatenating into FA's packed layout.
    m_block_offset = 0
    block_idx_offset = 0
    for b, (num_m_blocks_b, num_n_blocks_b) in enumerate(
        zip(num_m_blocks_per_seq, num_n_blocks_per_seq)
    ):
        partial_b = tile_partial[b, :num_m_blocks_b, :num_n_blocks_b]
        full_b = tile_full[b, :num_m_blocks_b, :num_n_blocks_b]
        next_m_block_offset = m_block_offset + num_m_blocks_b
        next_block_idx_offset = block_idx_offset + num_m_blocks_b * num_n_blocks_b

        mask_block_cnt[0, m_block_offset:next_m_block_offset] = partial_b.sum(dim=1).to(
            torch.int32
        )
        sort_keys = (~partial_b).to(torch.int32)
        mask_block_idx[0, block_idx_offset:next_block_idx_offset] = (
            sort_keys.argsort(dim=1, stable=True).to(torch.int32).flatten()
        )
        full_block_cnt[0, m_block_offset:next_m_block_offset] = full_b.sum(dim=1).to(
            torch.int32
        )
        sort_keys = (~full_b).to(torch.int32)
        full_block_idx[0, block_idx_offset:next_block_idx_offset] = (
            sort_keys.argsort(dim=1, stable=True).to(torch.int32).flatten()
        )
        cu_total_m_blocks.append(cu_total_m_blocks[-1] + num_m_blocks_b)
        cu_block_idx_offsets.append(
            cu_block_idx_offsets[-1] + num_m_blocks_b * num_n_blocks_b
        )
        m_block_offset = next_m_block_offset
        block_idx_offset = next_block_idx_offset

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        cu_total_m_blocks=torch.tensor(
            cu_total_m_blocks, dtype=torch.int32, device=tile_partial.device
        ),
        cu_block_idx_offsets=torch.tensor(
            cu_block_idx_offsets, dtype=torch.int32, device=tile_partial.device
        ),
        block_size=(tile_m, tile_n),
    )


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
    batch_idx = utils.ssa_to_scalar(batch)
    q_idx = utils.ssa_to_scalar(q_idx)
    kv_idx = utils.ssa_to_scalar(kv_idx)
    word_idx = kv_idx >> 5
    bit_idx = cutlass.Uint32(kv_idx & 31)
    word = dense_mask[batch_idx, q_idx, word_idx]
    result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
    result[0] = utils.shr_u32(cutlass.Uint32(word), bit_idx)
    return result.load()


dense_mask_mod.__vec_size__ = 32


def dense_mask_to_block_sparse(
    dense_mask: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seq_lens_q: Sequence[int] | None = None,
    seq_lens_k: Sequence[int] | None = None,
    num_heads: int = 1,
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
    pad_words = padded_words - dense_mask.shape[2]
    pad_q = padded_q - dense_mask.shape[1]
    if pad_words or pad_q:
        dense_mask = F.pad(dense_mask, (0, pad_words, 0, pad_q))
    dense_mask = dense_mask.reshape(
        batch_size,
        num_m_blocks,
        tile_m,
        num_n_blocks,
        words_per_tile,
    )
    tile_active = (dense_mask != 0).any(dim=4).any(dim=2)
    tile_full = (dense_mask == -1).all(dim=4).all(dim=2)
    tile_partial = tile_active & ~tile_full

    if seq_lens_q is not None:
        if seq_lens_k is None:
            seq_lens_k = seq_lens_q
        return _tiles_to_varlen_block_sparse(
            tile_partial,
            tile_full,
            seq_lens_q,
            seq_lens_k,
            tile_m,
            tile_n,
        )

    mask_block_cnt = tile_partial.sum(dim=2).unsqueeze(1).to(torch.int32)
    sort_keys = (~tile_partial).to(torch.int32)
    mask_block_idx = sort_keys.argsort(dim=2, stable=True).to(torch.int32)
    mask_block_idx = mask_block_idx.unsqueeze(1)
    full_block_cnt = tile_full.sum(dim=2).unsqueeze(1).to(torch.int32)
    sort_keys = (~tile_full).to(torch.int32)
    full_block_idx = sort_keys.argsort(dim=2, stable=True).to(torch.int32)
    full_block_idx = full_block_idx.unsqueeze(1)

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(tile_m, tile_n),
    )
