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
    word_idx = kv_idx >> 5
    bit_idx = cutlass.Uint32(kv_idx & 31)
    word = dense_mask[batch_idx, q_idx, word_idx]
    result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
    result[0] = utils.shr_u32(cutlass.Uint32(word), bit_idx)
    return result.load()


dense_mask_mod.__vec_size__ = 32


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
