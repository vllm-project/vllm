# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Contains replacement functions to fallback Triton usages in CPU backend
"""

from collections.abc import Callable

import torch


class _FuncWrapper:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __getitem__(self, *args, **kwargs) -> Callable:
        return self.func


# For _compute_slot_mapping_kernel in vllm/v1/worker/block_table.py
def _compute_slot_mapping_kernel_impl(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,  # [num_reqs + 1], int32
    positions: torch.Tensor,  # [num_tokens], int64
    block_table: torch.Tensor,  # [max_num_reqs, max_num_blocks_per_req], int32
    block_table_stride: int,  # max_num_blocks_per_req
    block_size: int,
    slot_mapping: torch.Tensor,  # [max_num_tokens], int64
    TOTAL_CP_WORLD_SIZE: int,
    TOTAL_CP_RANK: int,
    CP_KV_CACHE_INTERLEAVE_SIZE: int,
    PAD_ID: int,
    BLOCK_SIZE: int,
) -> None:
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on CPU."
    torch.ops._C.compute_slot_mapping_kernel_impl(
        query_start_loc,
        positions,
        block_table,
        slot_mapping,
        block_size,
    )


compute_slot_mapping_kernel = _FuncWrapper(_compute_slot_mapping_kernel_impl)


def _ensure_int64(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.int64 else t.to(torch.int64)


def _eagle_prepare_inputs_padded_kernel_impl(
    cu_num_draft_tokens,
    valid_sampled_tokens_count,
    query_start_loc_gpu,
    token_indices_to_sample,
    num_rejected_tokens_gpu,
    num_reqs,
):
    # C++ expects int64 for cu_num_draft_tokens, valid_sampled_tokens_count,
    # and num_rejected_tokens_gpu, but Python allocates them as int32.
    orig_rejected_dtype = num_rejected_tokens_gpu.dtype
    rejected_i64 = (
        num_rejected_tokens_gpu
        if orig_rejected_dtype == torch.int64
        else num_rejected_tokens_gpu.to(torch.int64)
    )
    torch.ops._C.eagle_prepare_inputs_padded_kernel_impl(
        _ensure_int64(cu_num_draft_tokens),
        _ensure_int64(valid_sampled_tokens_count),
        query_start_loc_gpu,
        token_indices_to_sample,
        rejected_i64,
        num_reqs,
    )
    if orig_rejected_dtype != torch.int64:
        num_rejected_tokens_gpu.copy_(rejected_i64.to(orig_rejected_dtype))


def _eagle_prepare_next_token_padded_kernel_impl(
    sampled_token_ids,
    discard_request_mask,
    backup_next_token_ids,
    next_token_ids,
    valid_sampled_tokens_count,
    vocab_size,
    num_sampled_tokens_per_req,
    num_reqs,
    stride=None,
    BLOCK_SIZE_TOKENS=None,
):
    # C++ reads all integer tensors as int64_t*. Output tensors are written
    # in-place so we create int64 copies, call C++, and copy back.
    orig_next_dtype = next_token_ids.dtype
    orig_valid_dtype = valid_sampled_tokens_count.dtype
    next_i64 = _ensure_int64(next_token_ids)
    valid_i64 = _ensure_int64(valid_sampled_tokens_count)
    torch.ops._C.eagle_prepare_next_token_padded_kernel_impl(
        _ensure_int64(sampled_token_ids),
        discard_request_mask,
        _ensure_int64(backup_next_token_ids),
        next_i64,
        valid_i64,
        vocab_size,
        num_sampled_tokens_per_req,
        num_reqs,
    )
    if orig_next_dtype != torch.int64:
        next_token_ids.copy_(next_i64.to(orig_next_dtype))
    if orig_valid_dtype != torch.int64:
        valid_sampled_tokens_count.copy_(valid_i64.to(orig_valid_dtype))


def _eagle_step_slot_mapping_metadata_kernel_impl(
    positions,
    block_table,
    stride,
    seq_lens,
    out_clamped_positions,
    out_slot_mapping,
    block_size,
    max_model_len,
    n_blocks_per_req,
    PAD_ID,
    batch_size=None,
):
    assert batch_size is None or batch_size == positions.shape[0], (
        f"batch_size mismatch: {batch_size} vs positions.shape[0]={positions.shape[0]}"
    )
    torch.ops._C.eagle_step_slot_mapping_metadata_kernel_impl(
        positions,
        block_table,
        seq_lens,
        out_clamped_positions,
        out_slot_mapping,
        block_size,
        max_model_len,
        PAD_ID,
    )


def _copy_and_expand_eagle_inputs_kernel_impl(
    target_token_ids_ptr,
    target_positions_ptr,
    next_token_ids_ptr,
    out_input_ids_ptr,
    out_positions_ptr,
    out_is_rejected_token_mask_ptr,
    out_is_masked_token_mask_ptr,
    out_new_token_indices_ptr,
    out_hidden_state_mapping_ptr,
    query_start_loc_ptr,
    query_end_loc_ptr,
    padding_token_id,
    parallel_drafting_token_id,
    total_input_tokens,
    num_padding_slots_per_request,
    shift_input_ids,
    BLOCK_SIZE_TOKENS=None,
    BLOCK_SIZE_REQS=None,
):
    """Adapter between Triton kernel call convention and C++ implementation.

    The Triton kernel uses '_ptr' suffixed parameter names and compile-time
    constants (BLOCK_SIZE_TOKENS, BLOCK_SIZE_REQS) which are not needed by
    the C++ implementation. C++ reads token id tensors as int64_t*.
    Output tensors that are int32 need copy-back after C++ writes int64.
    """
    orig_ids_dtype = out_input_ids_ptr.dtype
    orig_pos_dtype = out_positions_ptr.dtype
    out_ids_i64 = _ensure_int64(out_input_ids_ptr)
    out_pos_i64 = _ensure_int64(out_positions_ptr)
    torch.ops._C.copy_and_expand_eagle_inputs_kernel_impl(
        _ensure_int64(target_token_ids_ptr),
        _ensure_int64(target_positions_ptr),
        _ensure_int64(next_token_ids_ptr),
        out_ids_i64,
        out_pos_i64,
        out_is_rejected_token_mask_ptr,
        out_is_masked_token_mask_ptr,
        out_new_token_indices_ptr,
        out_hidden_state_mapping_ptr,
        query_start_loc_ptr,
        query_end_loc_ptr,
        padding_token_id,
        parallel_drafting_token_id,
        total_input_tokens,
        num_padding_slots_per_request,
        shift_input_ids,
    )
    if orig_ids_dtype != torch.int64:
        out_input_ids_ptr.copy_(out_ids_i64.to(orig_ids_dtype))
    if orig_pos_dtype != torch.int64:
        out_positions_ptr.copy_(out_pos_i64.to(orig_pos_dtype))


def _rejection_greedy_sample_kernel_impl(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    is_greedy,
    max_spec_len,
    uniform_probs=None,
    synthetic_conditional_rates=None,
    SYNTHETIC_MODE=False,
):
    # C++ kernel expects int64 for all integer tensors.
    # Note: uniform_probs, synthetic_conditional_rates, and SYNTHETIC_MODE are
    # passed by the rejection sampler for synthetic mode support, but are not
    # yet implemented in the C++ CPU kernel. We accept them here to maintain
    # compatibility with the kernel calling convention.
    assert not SYNTHETIC_MODE, "Synthetic acceptance not supported with CPU sampling"
    orig_dtype = output_token_ids.dtype
    output_token_ids_i64 = _ensure_int64(output_token_ids)
    torch.ops._C.rejection_greedy_sample_kernel_impl(
        output_token_ids_i64,
        _ensure_int64(cu_num_draft_tokens),
        _ensure_int64(draft_token_ids),
        _ensure_int64(target_argmax),
        _ensure_int64(bonus_token_ids),
        is_greedy,
        max_spec_len,
    )
    if orig_dtype != torch.int64:
        output_token_ids.copy_(output_token_ids_i64.to(orig_dtype))


def _rejection_random_sample_kernel_impl(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    bonus_token_ids,
    recovered_token_ids,
    uniform_probs,
    is_greedy,
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates=None,
    NO_DRAFT_PROBS=False,
    SYNTHETIC_MODE=False,
):
    # C++ kernel expects int64 for all integer tensors and float32 for probs.
    # uniform_probs is intentionally float64 in Python to avoid exact-zero
    # samples; cast to float32 here for C++ compatibility.
    # Note: synthetic_conditional_rates and SYNTHETIC_MODE are passed by the
    # rejection sampler for synthetic mode support, but are not yet implemented
    # in the C++ CPU kernel. We accept them here to maintain compatibility with
    # the kernel calling convention.
    assert not SYNTHETIC_MODE, "Synthetic acceptance not supported with CPU sampling"
    orig_dtype = output_token_ids.dtype
    output_token_ids_i64 = _ensure_int64(output_token_ids)
    torch.ops._C.rejection_random_sample_kernel_impl(
        output_token_ids_i64,
        _ensure_int64(cu_num_draft_tokens),
        _ensure_int64(draft_token_ids),
        draft_probs,
        target_probs,
        _ensure_int64(bonus_token_ids),
        _ensure_int64(recovered_token_ids),
        uniform_probs.to(torch.float32),
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS,
    )
    if orig_dtype != torch.int64:
        output_token_ids.copy_(output_token_ids_i64.to(orig_dtype))


def _expand_kernel_impl(
    output,
    input_val,
    cu_num_tokens,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS=None,
):
    torch.ops._C.expand_kernel_impl(
        _ensure_int64(output),
        _ensure_int64(input_val),
        _ensure_int64(cu_num_tokens),
        replace_from,
        replace_to,
    )


def _sample_recovered_tokens_kernel_impl(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    inv_q,
    vocab_size,
    BLOCK_SIZE=None,
    NO_DRAFT_PROBS=False,
):
    # C++ reads integer tensors as int64_t*; ensure correct dtype.
    orig_dtype = output_token_ids.dtype
    output_i64 = _ensure_int64(output_token_ids)
    torch.ops._C.sample_recovered_tokens_kernel_impl(
        output_i64,
        _ensure_int64(cu_num_draft_tokens),
        _ensure_int64(draft_token_ids),
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        NO_DRAFT_PROBS,
    )
    if orig_dtype != torch.int64:
        output_token_ids.copy_(output_i64.to(orig_dtype))


eagle_prepare_inputs_padded_kernel = _FuncWrapper(
    _eagle_prepare_inputs_padded_kernel_impl
)
eagle_prepare_next_token_padded_kernel = _FuncWrapper(
    _eagle_prepare_next_token_padded_kernel_impl
)
copy_and_expand_eagle_inputs_kernel = _FuncWrapper(
    _copy_and_expand_eagle_inputs_kernel_impl
)
eagle_step_slot_mapping_metadata_kernel = _FuncWrapper(
    _eagle_step_slot_mapping_metadata_kernel_impl
)
rejection_greedy_sample_kernel = _FuncWrapper(_rejection_greedy_sample_kernel_impl)
rejection_random_sample_kernel = _FuncWrapper(_rejection_random_sample_kernel_impl)
expand_kernel = _FuncWrapper(_expand_kernel_impl)
sample_recovered_tokens_kernel = _FuncWrapper(_sample_recovered_tokens_kernel_impl)
