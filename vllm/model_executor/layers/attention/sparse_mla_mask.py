# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cutlass
import cutlass.cute as cute

from vllm.vllm_flash_attn.cute import utils


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
    batch_stride, query_stride, _ = dense_mask.stride
    aligned_mask = cute.make_tensor(
        dense_mask.iterator,
        cute.make_layout(
            dense_mask.shape,
            stride=(
                cute.assume(batch_stride, divby=4),
                cute.assume(query_stride, divby=4),
                1,
            ),
        ),
    )
    mask_row = aligned_mask[batch_idx, q_idx, None]
    mask_chunks = cute.flat_divide(mask_row, (4,))
    mask_chunk = mask_chunks[None, (kv_idx >> 5) >> 2]
    loaded = cute.make_rmem_tensor_like(mask_chunk)
    cute.autovec_copy(mask_chunk, loaded)
    result = cute.make_rmem_tensor(4, dtype=cutlass.Uint32)
    for i in cutlass.range_constexpr(4):
        result[i] = cutlass.Uint32(loaded[i])
    return result.load()


dense_mask_mod.__vec_size__ = 128


@cute.jit
def offset_dense_mask_mod(
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
    key_start = dense_mask[batch_idx, 0, dense_mask.shape[2] - 1]
    kv_idx = utils.ssa_to_scalar(kv_idx) + key_start
    word_idx = kv_idx >> 5
    bit_idx = cutlass.Uint32(kv_idx & 31)
    word = dense_mask[batch_idx, q_idx, word_idx]
    result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
    result[0] = utils.shr_u32(cutlass.Uint32(word), bit_idx)
    return result.load()


offset_dense_mask_mod.__vec_size__ = 32
