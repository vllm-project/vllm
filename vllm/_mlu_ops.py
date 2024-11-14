from typing import List, Optional, Tuple

import torch

import math
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import torch_mlu_ops as tmo
except ImportError as e:
    logger.warning("Failed to import from torch_mlu_ops with %r", e)


def fused_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    store_output_before_norm: bool,
    quant_scale: torch.Tensor = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return tmo.fused_rms_norm(
                x, residual, gamma, beta, bias,
                eps, store_output_before_norm, quant_scale)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_kv: torch.Tensor,
    alibi_slope: torch.Tensor,
    attn_bias: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    compute_dtype: torch.dtype = torch.float,
    return_lse: bool = False,
    block_tables: torch.Tensor = None,
    k_cache_qunat_scale: torch.Tensor = None,
    v_cache_quant_scale: torch.Tensor = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return tmo.flash_attention(
        q, k, v, out,
        cu_seq_lens_q, cu_seq_lens_kv,
        alibi_slope, attn_bias,
        max_seq_len_q, max_seq_len_kv,
        softmax_scale, is_causal,
        window_size_left, window_size_right,
        compute_dtype, return_lse,
        block_tables, k_cache_qunat_scale,
        v_cache_quant_scale)


def single_query_cached_kv_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    k_cache_quant_scale: Optional[torch.Tensor],
    v_cache_quant_scale: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    max_contxt_len: int,
    windows_size_left: int,
    windows_size_right: int,
    softmax_scale: float,
) -> None:
    tmo.single_query_cached_kv_attn(
        q, k_cache, v_cache, out,
        block_tables, context_lens,
        k_cache_quant_scale, v_cache_quant_scale,
        alibi_slopes, max_contxt_len,
        windows_size_left, windows_size_right, softmax_scale)


def reshape_paged_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
) -> None:
    tmo.reshape_paged_cache(k, v, k_cache, v_cache, slot_mapping)


def swap_blocks(
    dst: torch.Tensor,
    src: torch.Tensor,
    block_mapping: torch.Tensor
) -> None:
    # FIXME: Remove this conversion after
    # tmo.swap_blocks support block_mapping tensor.
    block_mapping = block_mapping.tolist()
    block_mapping = {src: dst for src, dst in block_mapping}
    return tmo.swap_blocks(dst, src, block_mapping)


def copy_blocks(
    k_caches: List[torch.Tensor],
    v_caches: List[torch.Tensor],
    block_mapping: torch.Tensor
) -> None:
    # FIXME: Remove this conversion after
    # tmo.swap_blocks support block_mapping tensor.
    block_mapping = block_mapping.tolist()
    result_dict = {}
    for row in block_mapping:
        key = row[0]
        values = row[1:]
        if key in result_dict:
            result_dict[key].extend(values)
        else:
            result_dict[key] = values
    return tmo.copy_blocks(k_caches, v_caches, result_dict)


def advance_step(num_seqs: int,
                 num_queries: int,
                 block_size: int,
                 input_tokens: torch.Tensor,
                 sampled_token_ids: torch.Tensor,
                 input_positions: torch.Tensor,
                 seq_lens: torch.Tensor,
                 slot_mapping: torch.Tensor,
                 block_tables: torch.Tensor,
                 TILE_SIZE: int = 64) -> None:
    """
    Advance a step on MLU for existing inputs for a multi-step runner, which
    will update input_tokens/seq_lens/input_positions/slot_mapping inplace.
    """
    def verify_tensor(
        name: str,
        tensor: torch.Tensor,
        size_0: int,
        size_1: int,
        dtype: torch.dtype,
    ):
        """
        Auxiliary function to check whether input is valid.
        """
        size_0_cond = (size_0 == -1 or tensor.size(0) == size_0)
        size_1_cond = (size_1 == -1 or tensor.size(1) == size_1)
        if not (size_0_cond and size_1_cond and tensor.is_contiguous and tensor.dtype == dtype):
            raise ValueError(
                f"The input to advance_step is invalid with tensor name = {name}, "
                f"shape = {tensor.shape}, "
                f"is_cont = {tensor.is_contiguous()}, "
                f"type = {tensor.dtype}, "
                f"is not as expected: shape[{size_0}, {size_1}], type = {dtype}"
            )


    @triton.jit
    def _triton_advance_step(input_tokens_ptr,
                             sampled_token_ids_ptr,
                             input_positions_ptr,
                             seq_lens_ptr,
                             slot_mapping_ptr,
                             block_tables_ptr,
                             block_tables_stride,
                             num_seqs,
                             num_queries,
                             block_size,
                             TILE_SIZE: tl.constexpr,
    ):
        """
        The triton implementation of advance step.
        Reference: https://github.com/vllm-project/vllm/blob/v0.6.1/csrc/prepare_inputs/advance_step.cu#L14-L55
        """
        # Set meta info.
        pid = tl.program_id(axis=0)
        offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < num_queries

        # Update input_tokens.
        sampled_token_ids = tl.load(sampled_token_ids_ptr + offsets, mask=mask)
        tl.store(input_tokens_ptr + offsets, sampled_token_ids, mask=mask)

        seq_lens = tl.load(seq_lens_ptr + offsets, mask=mask)
        next_seq_lens = seq_lens + 1
        next_input_pos = next_seq_lens - 1

        # Update seq_lens.
        tl.store(seq_lens_ptr + offsets, next_seq_lens, mask=mask)

        # Update input_positions.
        tl.store(input_positions_ptr + offsets, next_input_pos, mask=mask)

        # Calculate slot num.
        block_index = next_input_pos // block_size
        block_offset = next_input_pos % block_size
        block_tables = tl.load(block_tables_ptr + block_tables_stride * offsets + block_index, mask=mask)
        slot_num = block_tables * block_size + block_offset

        # Update slot_mapping.
        tl.store(slot_mapping_ptr + offsets, slot_num, mask=mask)


    verify_tensor("input_tokens", input_tokens, num_seqs, -1, torch.int64)
    verify_tensor("sampled_token_ids", sampled_token_ids, num_queries, 1, torch.int64)
    verify_tensor("input_positions", input_positions, num_seqs, -1, torch.int32)
    verify_tensor("seq_lens", seq_lens, num_seqs, -1, torch.int32)
    verify_tensor("slot_mapping", slot_mapping, num_seqs, -1, torch.int32)
    verify_tensor("block_tables", block_tables, num_seqs, -1, torch.int32)

    grid = (math.ceil(num_queries / TILE_SIZE), )
    _triton_advance_step[grid](input_tokens,
                               sampled_token_ids,
                               input_positions,
                               seq_lens,
                               slot_mapping,
                               block_tables,
                               block_tables.stride(0),
                               num_seqs,
                               num_queries,
                               block_size,
                               TILE_SIZE)
