# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

PADDING_SLOT_ID = -1


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


class EagleStepSlotMappingMetadataKernel(
    VllmJitKernel["EagleStepSlotMappingMetadataKernel.CompileKey"]
):
    """VllmJitKernel wrapper for ``eagle_step_slot_mapping_metadata_kernel``.

    At runtime ``block_table_stride == n_blocks_per_req``, so the stride's
    ==1 branch is covered by ``n_blocks_per_req == 1``.
    """

    _HYBRID_FACTORS = (1, 2, 4)

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        max_model_len: int
        n_blocks_per_req: int
        PAD_ID: int
        batch_size: int
        block_table_stride: int

    @staticmethod
    @triton.jit
    def kernel(
        positions_ptr,
        block_table_ptr,
        block_table_stride,
        seq_lens_ptr,
        out_clamped_positions_ptr,
        out_slot_mapping_ptr,
        block_size: tl.constexpr,
        max_model_len: tl.constexpr,
        n_blocks_per_req: tl.constexpr,
        PAD_ID: tl.constexpr,
        batch_size,
    ):
        req_idx = tl.program_id(0)

        if req_idx >= batch_size:
            tl.store(out_slot_mapping_ptr + req_idx, PAD_ID)
            return

        position = tl.load(positions_ptr + req_idx)
        new_position = position + 1

        exceeds_max = new_position >= max_model_len
        clamped_position = tl.where(exceeds_max, 0, new_position)

        block_number = clamped_position // block_size
        block_number = tl.minimum(block_number, n_blocks_per_req - 1)

        block_id = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_number
        )
        slot_id = block_id * block_size + (clamped_position % block_size)
        slot_id = tl.where(exceeds_max, PAD_ID, slot_id)

        seq_len = tl.load(seq_lens_ptr + req_idx)
        new_seq_len = tl.where(exceeds_max, 1, seq_len + 1)
        new_seq_len = tl.minimum(new_seq_len, max_model_len)

        tl.store(out_clamped_positions_ptr + req_idx, clamped_position)
        tl.store(out_slot_mapping_ptr + req_idx, slot_id)
        tl.store(seq_lens_ptr + req_idx, new_seq_len)

    def dispatch(  # type: ignore[override]
        self,
        *,
        block_size: int,
        max_model_len: int,
        n_blocks_per_req: int,
        PAD_ID: int,
        batch_size: int,
    ) -> CompileKey:
        return self.CompileKey(
            block_size=block_size,
            max_model_len=max_model_len,
            n_blocks_per_req=n_blocks_per_req,
            PAD_ID=PAD_ID,
            batch_size=batch_size,
            block_table_stride=n_blocks_per_req,
        )

    def get_warmup_keys(
        self,
        vllm_config: VllmConfig,
        *,
        block_size: int,
        max_model_len: int,
    ) -> list[CompileKey]:
        parallel_config = vllm_config.parallel_config
        total_cp_size = (
            parallel_config.decode_context_parallel_size
            * parallel_config.prefill_context_parallel_size
        )
        cp_factors = {1, total_cp_size}
        n_blocks_candidates: set[int] = set()
        for cp in cp_factors:
            base = cdiv(max_model_len, block_size * cp)
            for hybrid in self._HYBRID_FACTORS:
                n_blocks_candidates.add(max(base * hybrid, 1))

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        return self._trace_dispatch(self.dispatch)(
            block_size=block_size,
            max_model_len=max_model_len,
            n_blocks_per_req=sorted(n_blocks_candidates),
            PAD_ID=PADDING_SLOT_ID,
            batch_size=[1, max_num_seqs],
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int64_ptr = TritonWarmupTensor(torch.int64)
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            int64_ptr,  # positions_ptr
            int32_ptr,  # block_table_ptr
            compile_key.block_table_stride,
            int32_ptr,  # seq_lens_ptr
            int64_ptr,  # out_clamped_positions_ptr
            int64_ptr,  # out_slot_mapping_ptr
            block_size=compile_key.block_size,
            max_model_len=compile_key.max_model_len,
            n_blocks_per_req=compile_key.n_blocks_per_req,
            PAD_ID=compile_key.PAD_ID,
            batch_size=compile_key.batch_size,
            grid=(1,),
        )

    def __call__(
        self,
        positions_1d: torch.Tensor,
        block_table_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        max_model_len: int,
        out_clamped_positions: torch.Tensor,
        out_slot_mapping: torch.Tensor,
        input_batch_size: int | None = None,
    ) -> None:
        batch_size = positions_1d.shape[0]
        if input_batch_size is None:
            input_batch_size = batch_size
        n_blocks_per_req = block_table_tensor.shape[1]
        self.kernel[(input_batch_size,)](
            positions_1d,
            block_table_tensor,
            block_table_tensor.stride(0),
            seq_lens,
            out_clamped_positions,
            out_slot_mapping,
            block_size=block_size,
            max_model_len=max_model_len,
            n_blocks_per_req=n_blocks_per_req,
            PAD_ID=PADDING_SLOT_ID,
            batch_size=batch_size,
        )


_EAGLE_STEP_SLOT_MAPPING_KERNEL = EagleStepSlotMappingMetadataKernel()


def eagle_step_update_slot_mapping_and_metadata(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
    out_clamped_positions: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    input_batch_size: int | None = None,
) -> None:
    _EAGLE_STEP_SLOT_MAPPING_KERNEL(
        positions_1d,
        block_table_tensor,
        seq_lens,
        block_size,
        max_model_len,
        out_clamped_positions,
        out_slot_mapping,
        input_batch_size,
    )


class EaglePrepareInputsPaddedKernel(
    VllmJitKernel["EaglePrepareInputsPaddedKernel.CompileKey"]
):
    """VllmJitKernel wrapper for ``eagle_prepare_inputs_padded_kernel``."""

    @dataclass(frozen=True)
    class CompileKey:
        num_reqs: int

    @staticmethod
    @triton.jit
    def kernel(
        cu_num_draft_tokens_ptr,
        valid_sampled_tokens_count_ptr,
        query_start_loc_gpu_ptr,
        token_indices_to_sample_ptr,
        num_rejected_tokens_gpu_ptr,
        num_reqs,
    ):
        req_idx = tl.program_id(axis=0)
        if req_idx >= num_reqs:
            return

        cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)

        if req_idx == 0:
            num_draft_tokens = cu_draft_curr
        else:
            cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
            num_draft_tokens = cu_draft_curr - cu_draft_prev

        valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
        num_rejected_tokens = num_draft_tokens + 1 - valid_count
        num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

        q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1

        index_to_sample = q_last_tok_idx - num_rejected_tokens
        tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)
        tl.store(num_rejected_tokens_gpu_ptr + req_idx, num_rejected_tokens)

    def dispatch(self, *, num_reqs: int) -> CompileKey:  # type: ignore[override]
        return self.CompileKey(num_reqs=num_reqs)

    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        return self._trace_dispatch(self.dispatch)(
            num_reqs=[1, max_num_seqs],
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            int32_ptr,
            int32_ptr,
            int32_ptr,
            int32_ptr,
            int32_ptr,
            compile_key.num_reqs,
            grid=(1,),
        )

    def __call__(
        self,
        cu_num_draft_tokens: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
        query_start_loc: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor,
        num_reqs: int,
    ) -> None:
        self.kernel[(num_reqs,)](
            cu_num_draft_tokens,
            valid_sampled_tokens_count,
            query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )


_EAGLE_PREPARE_INPUTS_KERNEL = EaglePrepareInputsPaddedKernel()


class EaglePrepareNextTokenPaddedKernel(
    VllmJitKernel["EaglePrepareNextTokenPaddedKernel.CompileKey"]
):
    """VllmJitKernel wrapper for ``eagle_prepare_next_token_padded_kernel``.

    At runtime ``stride_sampled_token_ids == num_sampled_tokens_per_req``
    (contiguous tensor), so the stride's ==1 branch is covered by
    ``num_sampled_tokens_per_req == 1``.
    """

    @dataclass(frozen=True)
    class CompileKey:
        BLOCK_SIZE_TOKENS: int
        vocab_size: int
        num_sampled_tokens_per_req: int
        num_reqs: int
        stride_sampled_token_ids: int

    @staticmethod
    @triton.jit
    def kernel(
        sampled_token_ids_ptr,
        discard_request_mask_ptr,
        backup_next_token_ids_ptr,
        next_token_ids_ptr,
        valid_sampled_tokens_count_ptr,
        vocab_size,
        num_sampled_tokens_per_req,
        num_reqs,
        stride_sampled_token_ids,
        BLOCK_SIZE_TOKENS: tl.constexpr,
    ):
        req_idx = tl.program_id(axis=0)
        if req_idx >= num_reqs:
            return

        is_discarded = tl.load(discard_request_mask_ptr + req_idx)

        if is_discarded:
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            valid_count = tl.full((), 0, dtype=tl.uint32)
            tl.store(next_token_ids_ptr + req_idx, backup_token)
            tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
        else:
            token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
            token_mask = token_offs < num_sampled_tokens_per_req

            row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
            token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

            is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
            valid_count = tl.sum(is_valid_mask)

            if valid_count > 0:
                last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))
                last_valid_token = tl.sum(
                    tl.where(token_offs == last_valid_index, token_ids, 0)
                )
                tl.store(next_token_ids_ptr + req_idx, last_valid_token)
            else:
                backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
                tl.store(next_token_ids_ptr + req_idx, backup_token)

            tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)

    def dispatch(  # type: ignore[override]
        self,
        *,
        BLOCK_SIZE_TOKENS: int,
        vocab_size: int,
        num_sampled_tokens_per_req: int,
        num_reqs: int,
    ) -> CompileKey:
        return self.CompileKey(
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
            vocab_size=vocab_size,
            num_sampled_tokens_per_req=num_sampled_tokens_per_req,
            num_reqs=num_reqs,
            stride_sampled_token_ids=num_sampled_tokens_per_req,
        )

    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        spec_config = vllm_config.speculative_config
        num_spec_tokens = getattr(spec_config, "num_speculative_tokens", None)
        if num_spec_tokens is None or num_spec_tokens <= 0:
            return []
        num_sampled = num_spec_tokens + 1
        vocab_size = vllm_config.model_config.get_vocab_size()
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        # Cover all powers of two up to next_power_of_2(num_sampled) so the
        # cache also covers configurations with smaller num_speculative_tokens.
        max_block = next_power_of_2(num_sampled)
        block_sizes = [1]
        b = 2
        while b <= max_block:
            block_sizes.append(b)
            b *= 2
        return self._trace_dispatch(self.dispatch)(
            BLOCK_SIZE_TOKENS=block_sizes,
            vocab_size=vocab_size,
            num_sampled_tokens_per_req=[1, num_sampled],
            num_reqs=[1, max_num_seqs],
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int32_ptr = TritonWarmupTensor(torch.int32)
        bool_ptr = TritonWarmupTensor(torch.bool)
        warmup(
            int32_ptr,  # sampled_token_ids_ptr
            bool_ptr,  # discard_request_mask_ptr
            int32_ptr,  # backup_next_token_ids_ptr
            int32_ptr,  # next_token_ids_ptr
            int32_ptr,  # valid_sampled_tokens_count_ptr
            compile_key.vocab_size,
            compile_key.num_sampled_tokens_per_req,
            compile_key.num_reqs,
            compile_key.stride_sampled_token_ids,
            BLOCK_SIZE_TOKENS=compile_key.BLOCK_SIZE_TOKENS,
            grid=(1,),
        )

    def __call__(
        self,
        sampled_token_ids: torch.Tensor,
        discard_request_mask: torch.Tensor,
        backup_next_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
        vocab_size: int,
        num_sampled_tokens_per_req: int,
        num_reqs: int,
        stride_sampled_token_ids: int,
        BLOCK_SIZE_TOKENS: int,
    ) -> None:
        self.kernel[(num_reqs,)](
            sampled_token_ids,
            discard_request_mask,
            backup_next_token_ids,
            next_token_ids,
            valid_sampled_tokens_count,
            vocab_size,
            num_sampled_tokens_per_req,
            num_reqs,
            stride_sampled_token_ids,
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )


_EAGLE_PREPARE_NEXT_TOKEN_KERNEL = EaglePrepareNextTokenPaddedKernel()


def compute_new_slot_mapping(
    cad: CommonAttentionMetadata,
    new_positions: torch.Tensor,
    is_rejected_token_mask: torch.Tensor,
    block_size: int,
    num_new_tokens: int,
    max_model_len: int,
):
    batch_size, n_blocks_per_req = cad.block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=cad.query_start_loc.device)
    req_indices = torch.repeat_interleave(
        req_indices,
        cad.naive_query_lens() + num_new_tokens,
        output_size=len(new_positions),
    )
    # Clamp the positions to prevent an out-of-bounds error when indexing
    # into block_table_tensor.
    clamped_positions = torch.clamp(new_positions, max=max_model_len - 1)
    block_table_indices = (
        req_indices * n_blocks_per_req + clamped_positions // block_size
    )
    block_nums = cad.block_table_tensor.view(-1)[block_table_indices]
    block_offsets = clamped_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # Mask out the position ids that exceed the max model length.
    exceeds_max_model_len = new_positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
    # Mask out rejected tokens to prevent saves to the KV cache.
    new_slot_mapping.masked_fill_(is_rejected_token_mask, PADDING_SLOT_ID)
    return new_slot_mapping


def extend_all_queries_by_N(
    common_attn_metadata: CommonAttentionMetadata,
    N: int,
    arange: torch.Tensor,
    new_slot_mapping: torch.Tensor,
) -> CommonAttentionMetadata:
    """
    Creates a new CommonAttentionMetadata with all query lengths increased by N.
    Also all seq lens are increased by N.
    This is useful e.g. in speculative decoding with parallel drafting, where we
    extend each sequence by N tokens and predict all tokens in one pass.
    The slot mapping is computed externally, as it requires more information.
    """
    cad = common_attn_metadata
    # query start loc must be increased by [+0, +N, +2N, ..., +batch_size * N]
    new_query_start_loc = cad.query_start_loc + N * arange[: len(cad.query_start_loc)]
    new_query_start_loc_cpu = cad.query_start_loc_cpu + N * torch.arange(
        len(cad.query_start_loc_cpu), dtype=torch.int32
    )
    new_cad = cad.replace(
        query_start_loc=new_query_start_loc,
        query_start_loc_cpu=new_query_start_loc_cpu,
        seq_lens=cad.seq_lens + N,
        # each request is extended by N tokens -> batch_size * N tokens are added
        num_actual_tokens=cad.num_actual_tokens + cad.batch_size() * N,
        # All query lens increase by N, so max query len increases by N
        max_query_len=cad.max_query_len + N,
        max_seq_len=cad.max_seq_len + N,
        slot_mapping=new_slot_mapping,
    )
    return new_cad


# Unified copy/expand kernel
@triton.jit
def copy_and_expand_eagle_inputs_kernel(
    # (Padded) Inputs from the target model
    target_token_ids_ptr,  # [total_tokens_in_batch]
    target_positions_ptr,  # [total_tokens_in_batch]
    next_token_ids_ptr,  # [num_reqs]
    # Outputs to the drafting buffers
    out_input_ids_ptr,  # [total_draft_tokens_in_batch] (output)
    out_positions_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_rejected_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_masked_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_new_token_indices_ptr,  # [num_padding_slots_per_request * num_reqs] (output)
    out_hidden_state_mapping_ptr,  # [total_tokens_in_batch]
    # Input metadata
    query_start_loc_ptr,  # [num_reqs + 1], last value is the total num input tokens
    query_end_loc_ptr,  # [num_reqs]
    padding_token_id,  # tl.int32
    parallel_drafting_token_id,  # tl.int32
    # Sizing info
    total_input_tokens,  # tl.int32
    num_padding_slots_per_request,  # tl.int32
    shift_input_ids,  # tl.bool
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Blocks along token dim to handle prefills
):
    """
    Copy and expand inputs from the target model to the drafting buffers for Eagle
    speculative decoding. This kernel handles padding slots and parallel drafting
    tokens, if enabled.
    """
    request_idx = tl.program_id(axis=0)
    token_batch_idx = tl.program_id(axis=1)

    # Load query locations
    query_start_loc = tl.load(query_start_loc_ptr + request_idx)
    next_query_start_loc = tl.load(query_start_loc_ptr + request_idx + 1)
    query_end_loc = tl.load(query_end_loc_ptr + request_idx)

    # Calculate number of valid tokens to copy and input offset
    # With shift_input_ids=True, we skip the first token
    # Output layout: each request gets (input_len + num_padding_slots_per_request) slots
    # But with shift, we lose one token per request
    if shift_input_ids:
        num_valid_tokens = query_end_loc - query_start_loc
        input_offset = 1
        output_start = query_start_loc + request_idx * (
            num_padding_slots_per_request - 1
        )
    else:
        num_valid_tokens = query_end_loc - query_start_loc + 1
        input_offset = 0
        output_start = query_start_loc + request_idx * num_padding_slots_per_request

    # Number of rejected tokens from previous speculation
    num_rejected = next_query_start_loc - query_end_loc - 1

    # Total output tokens for this request
    total_output_tokens = (
        num_valid_tokens + num_padding_slots_per_request + num_rejected
    )

    # Process tokens in this block
    j = token_batch_idx * BLOCK_SIZE_TOKENS + tl.arange(0, BLOCK_SIZE_TOKENS)

    # Compute masks for different output regions:
    # [0, num_valid_tokens): valid tokens copied from input
    # [num_valid_tokens]: bonus token from next_token_ids
    # (num_valid_tokens, num_valid_tokens + num_padding_slots_per_request):
    #     parallel drafting slots
    # [num_valid_tokens + num_padding_slots_per_request, total_output_tokens):
    #     rejected slots
    in_bounds = j < total_output_tokens
    is_valid_region = j < num_valid_tokens
    is_bonus_region = j == num_valid_tokens
    is_parallel_draft_region = (j > num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    is_rejected_region = j >= num_valid_tokens + num_padding_slots_per_request

    # Compute output indices
    out_idx = output_start + j

    # For valid tokens, compute input index
    in_idx = query_start_loc + input_offset + j
    # Clamp to avoid out-of-bounds access (masked loads still need valid addresses)
    in_idx_clamped = tl.minimum(in_idx, total_input_tokens - 1)

    # Load input tokens (masked to valid region)
    token_ids = tl.load(
        target_token_ids_ptr + in_idx_clamped, mask=is_valid_region & in_bounds, other=0
    )

    # Load the starting position for this request (first position in the sequence)
    start_pos = tl.load(target_positions_ptr + query_start_loc)

    # Load bonus token for this request
    bonus_token = tl.load(next_token_ids_ptr + request_idx)

    # Build final token_ids based on region
    token_ids = tl.where(is_bonus_region, bonus_token, token_ids)
    token_ids = tl.where(
        is_parallel_draft_region, parallel_drafting_token_id, token_ids
    )
    token_ids = tl.where(is_rejected_region, padding_token_id, token_ids)

    # Build final positions:
    # Positions are NOT shifted - they start from the first input position and increment
    # Output position j gets start_pos + j
    # (e.g., input positions [5,6,7] -> output [5,6,7,8,9,...])
    positions = start_pos + j
    # Rejected positions are don't-care, set to 0
    positions = tl.where(is_rejected_region, 0, positions)

    # Compute output masks
    is_rejected_out = is_rejected_region & in_bounds
    is_masked_out = is_parallel_draft_region & in_bounds

    # Compute indices of new tokens (bonus + parallel drafting) for sampling
    # New tokens are at positions
    #     [num_valid_tokens, num_valid_tokens + num_padding_slots_per_request)
    is_new_token_region = (j >= num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    new_token_local_idx = (
        j - num_valid_tokens
    )  # 0 for bonus, 1, 2, ... for parallel drafting
    new_token_out_idx = (
        request_idx * num_padding_slots_per_request + new_token_local_idx
    )

    # Compute hidden state mapping (source index -> destination index)
    # This maps each input position to its corresponding output position
    # Hidden states don't get shifted, so we map all input tokens (including rejected)
    if shift_input_ids:
        num_input_tokens_this_request = next_query_start_loc - query_start_loc
        is_input_region = j < num_input_tokens_this_request
        src_idx = query_start_loc + j
        tl.store(out_hidden_state_mapping_ptr + src_idx, out_idx, mask=is_input_region)

    # Store outputs
    tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
    tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)
    tl.store(out_is_rejected_token_mask_ptr + out_idx, is_rejected_out, mask=in_bounds)
    tl.store(out_is_masked_token_mask_ptr + out_idx, is_masked_out, mask=in_bounds)
    tl.store(
        out_new_token_indices_ptr + new_token_out_idx,
        out_idx,
        mask=is_new_token_region & in_bounds,
    )


@triton.jit
def copy_and_expand_dflash_inputs_kernel(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total] (output)
    out_context_positions_ptr,  # [num_context] (output)
    out_query_positions_ptr,  # [num_query_total] (output)
    out_context_slot_mapping_ptr,  # [num_context] (output)
    out_query_slot_mapping_ptr,  # [num_query_total] (output)
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (in elements)
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    num_rejected_tokens_ptr,  # [num_reqs] or null (0) when not padded
    # Scalars
    parallel_drafting_token_id,  # tl.int32
    block_size,  # tl.int32
    num_query_per_req,  # tl.int32
    num_speculative_tokens,  # tl.int32
    total_input_tokens,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
    HAS_NUM_REJECTED: tl.constexpr = False,
):
    """
    Fused kernel for DFlash first-pass input setup.

    Per request, this kernel:
      1. Copies context positions from target_positions to
         out_context_positions.
      2. Computes query positions (last_target_pos + 1 + offset) and writes
         them to out_query_positions.
      3. Writes input_ids for query tokens: [next_token, mask, mask, ...].
      4. Computes slot_mapping for context and query positions into separate
         buffers via block_table lookup.
      5. Writes token_indices_to_sample for the mask (speculative) tokens.
    """
    req_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    # Load context token range for this request
    ctx_start = tl.load(query_start_loc_ptr + req_idx)
    ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    total_tokens = num_ctx + num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = j < total_tokens
    is_ctx = j < num_ctx
    is_query = (~is_ctx) & in_bounds
    query_off = j - num_ctx  # offset within query portion (0-indexed)

    # --- Positions ---
    # Context: load from target_positions
    ctx_pos_idx = tl.minimum(ctx_start + j, total_input_tokens - 1)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)

    # Query: last_valid_pos + 1 + query_off
    # In padded mode, ctx_end includes rejected tokens; use valid_ctx_end
    # to find the last accepted context position.
    if HAS_NUM_REJECTED:
        num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
        valid_ctx_end = ctx_end - num_rejected
    else:
        valid_ctx_end = ctx_end
    last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_pos = last_pos + 1 + query_off

    positions = tl.where(is_ctx, ctx_pos, query_pos)

    # Context and query positions go to separate buffers.
    ctx_pos_out = ctx_start + j
    tl.store(out_context_positions_ptr + ctx_pos_out, ctx_pos, mask=is_ctx)
    query_out = req_idx * num_query_per_req + query_off
    tl.store(out_query_positions_ptr + query_out, query_pos, mask=is_query)

    # --- Slot mapping (block_table lookup for all positions) ---
    block_num = positions // block_size
    # # Clamp block_number to avoid OOB when position is at max
    block_num = tl.minimum(block_num, block_table_stride - 1)
    block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_num,
        mask=in_bounds,
        other=0,
    ).to(tl.int64)
    slot = block_id * block_size + (positions % block_size)
    tl.store(out_context_slot_mapping_ptr + ctx_pos_out, slot, mask=is_ctx)
    tl.store(out_query_slot_mapping_ptr + query_out, slot, mask=is_query)

    # --- Input IDs (query tokens only) ---
    bonus_token = tl.load(next_token_ids_ptr + req_idx)
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
    tl.store(out_input_ids_ptr + query_out, input_id, mask=is_query)

    # --- Token indices to sample (mask tokens, skip the bonus token) ---
    is_sample = is_query & (query_off > 0)
    sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1)
    tl.store(
        out_token_indices_ptr + sample_out_idx,
        query_out,
        mask=is_sample,
    )


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def update_num_computed_tokens_for_batch_change(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Correct num_computed_tokens for async spec decode drift.

    Requests that had drafts: corrected = prev_gpu + valid_count.
    New requests or non-draft (e.g. prefills): use CPU value directly.
    """
    # Clamp because prev_positions can be -1 for new requests
    gather_indices = prev_positions.clamp(min=0)

    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    n = prev_positions.shape[0]
    num_computed_tokens[:n].copy_(
        torch.where(participating, corrected, cpu_num_computed_tokens)
    )
    num_accepted_tokens.copy_(
        torch.where(participating, valid_counts, num_accepted_tokens)
    )


def unconditional_to_conditional_rates(rates: list[float]) -> list[float]:
    """Convert per-position unconditional rates to per-position conditional
    rates for the early-terminating rejection loop (c_i = p_i / p_{i-1})."""
    return [p / q if q > 0.0 else 0.0 for p, q in zip(rates, [1.0, *rates[:-1]])]
