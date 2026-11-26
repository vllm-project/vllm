# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cutlass
import cutlass.cute as cute

from vllm.vllm_flash_attn.cute.utils import ssa_to_scalar


@cute.jit
def sdsa_mask_mod(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
    """FA4 mask_mod of the SDSA prefill core: query q_idx of a batch is prefill
    token seqlen_info.offset_q + q_idx, whose row of aux_tensors[0] is its
    bitmask; returns the bits of union columns kv_idx .. kv_idx + 127. FA4 also
    calls it for its tile's padding rows (q_idx >= seqlen_q, output dropped);
    those read the batch's last row, as the last batch's would lie past the mask."""
    q = cutlass.min(ssa_to_scalar(q_idx), seqlen_info.seqlen_q - 1)
    row = aux_tensors[0][seqlen_info.offset_q + q, None]
    words = cute.make_rmem_tensor(4, cutlass.Int32)
    cute.autovec_copy(cute.local_tile(row, (4,), (ssa_to_scalar(kv_idx) >> 7,)), words)
    return cute.recast_tensor(words, cutlass.Uint32).load()


sdsa_mask_mod.__vec_size__ = 128
