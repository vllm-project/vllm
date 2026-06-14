# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for the synthetic shapes used by ``dry_run_helper_kernels``.

These tests pin the input shapes our spec-decode warmup uses against the
actual Triton kernels in ``vllm/v1/spec_decode/utils.py``, so a future
kernel-signature change that breaks our warmup will fail here rather than
silently regressing TTFT (vllm-project/vllm#39790).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata,
    next_power_of_2,
    update_num_computed_tokens_for_batch_change,
)

pytest.importorskip("triton")
if not current_platform.is_cuda_alike() and not current_platform.is_xpu():
    pytest.skip(
        "CUDA/XPU required for spec-decode warmup smoke tests",
        allow_module_level=True,
    )

DEVICE = torch.device(current_platform.device_type)


@pytest.mark.parametrize("num_speculative_tokens", [1, 4, 8])
def test_warmup_shapes_eagle_prepare_kernels(num_speculative_tokens: int) -> None:
    """The two padded-batch kernels warmed by the base proposer must accept
    the synthetic shapes ``SpecDecodeBaseProposer.dry_run_helper_kernels``
    builds.
    """
    num_reqs = 1
    num_sampled_per_req = num_speculative_tokens + 1
    block_size_tokens = next_power_of_2(num_sampled_per_req)

    sampled_token_ids = torch.zeros(
        (num_reqs, num_sampled_per_req), dtype=torch.int64, device=DEVICE
    )
    discard_mask = torch.zeros(num_reqs, dtype=torch.bool, device=DEVICE)
    backup_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=DEVICE)
    valid_count_out = torch.empty(num_reqs, dtype=torch.int32, device=DEVICE)
    eagle_prepare_next_token_padded_kernel[(num_reqs,)](
        sampled_token_ids,
        discard_mask,
        backup_tokens,
        next_token_ids,
        valid_count_out,
        128,
        num_sampled_per_req,
        num_reqs,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=block_size_tokens,
    )

    cu_num_draft_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    valid_sampled_count = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    token_indices_to_sample = torch.empty(num_reqs, dtype=torch.int32, device=DEVICE)
    num_rejected_tokens_gpu = torch.empty(num_reqs, dtype=torch.int32, device=DEVICE)
    eagle_prepare_inputs_padded_kernel[(num_reqs,)](
        cu_num_draft_tokens,
        valid_sampled_count,
        query_start_loc,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
        num_reqs,
    )
    torch.accelerator.synchronize()


@pytest.mark.parametrize("num_speculative_tokens", [2, 4])
@pytest.mark.parametrize("shift_input_ids", [True, False])
def test_warmup_shapes_copy_expand_eagle(
    num_speculative_tokens: int, shift_input_ids: bool
) -> None:
    """``copy_and_expand_eagle_inputs_kernel`` must accept the shapes the
    base proposer warmup builds for ``needs_extra_input_slots=True``
    configs (parallel-drafting Eagle / DraftModel)."""
    num_reqs = 1
    # Mirror ``needs_extra_input_slots=True`` math: extra slots == k for
    # parallel drafting; net slots = k - (1 if shift else 0).
    extra_slots = num_speculative_tokens
    net_slots = extra_slots - (1 if shift_input_ids else 0)
    if net_slots <= 0:
        pytest.skip("config does not allocate extra slots")
    num_query_per_req = 1 + net_slots
    total_input = num_reqs * num_query_per_req
    total_output = num_reqs * (num_query_per_req + extra_slots)
    # Match production: BLOCK_SIZE keyed off max_query_len + net_new_slots
    # (which equals num_query_per_req for a first-decode request).
    block_size_tokens = min(256, next_power_of_2(num_query_per_req))

    target_token_ids = torch.zeros(total_input, dtype=torch.int32, device=DEVICE)
    target_positions = torch.zeros(total_input, dtype=torch.int64, device=DEVICE)
    next_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    out_input_ids = torch.zeros(total_output, dtype=torch.int32, device=DEVICE)
    out_positions = torch.zeros(total_output, dtype=torch.int64, device=DEVICE)
    out_is_rejected = torch.zeros(total_output, dtype=torch.bool, device=DEVICE)
    out_is_masked = torch.zeros(total_output, dtype=torch.bool, device=DEVICE)
    out_token_indices = torch.empty(
        num_reqs * extra_slots, dtype=torch.int32, device=DEVICE
    )
    out_hidden_state_mapping = torch.empty(
        total_input, dtype=torch.int32, device=DEVICE
    )
    qsl = (
        torch.arange(num_reqs + 1, dtype=torch.int32, device=DEVICE) * num_query_per_req
    )
    qel = qsl[1:] - 1

    num_blocks = (total_output + block_size_tokens - 1) // block_size_tokens
    copy_and_expand_eagle_inputs_kernel[(num_reqs, num_blocks)](
        target_token_ids_ptr=target_token_ids,
        target_positions_ptr=target_positions,
        next_token_ids_ptr=next_token_ids,
        out_input_ids_ptr=out_input_ids,
        out_positions_ptr=out_positions,
        out_is_rejected_token_mask_ptr=out_is_rejected,
        out_is_masked_token_mask_ptr=out_is_masked,
        out_new_token_indices_ptr=out_token_indices,
        out_hidden_state_mapping_ptr=out_hidden_state_mapping,
        query_start_loc_ptr=qsl,
        query_end_loc_ptr=qel,
        padding_token_id=0,
        parallel_drafting_token_id=0,
        total_input_tokens=total_input,
        num_padding_slots_per_request=extra_slots,
        shift_input_ids=shift_input_ids,
        BLOCK_SIZE_TOKENS=block_size_tokens,
    )
    torch.accelerator.synchronize()


def test_warmup_shapes_eagle_step_slot_mapping() -> None:
    """The Eagle per-step slot-mapping wrapper must accept the shapes
    ``EagleProposer.dry_run_helper_kernels`` builds for sequential Eagle
    with k>=2."""
    num_reqs = 1
    block_size = 16
    max_model_len = 4096
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size

    positions_1d = torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE)
    block_table = torch.zeros(
        (num_reqs, n_blocks_per_req), dtype=torch.int32, device=DEVICE
    )
    seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=DEVICE)
    out_clamped_positions = torch.empty(num_reqs, dtype=torch.int64, device=DEVICE)
    out_slot_mapping = torch.empty(num_reqs, dtype=torch.int64, device=DEVICE)
    eagle_step_update_slot_mapping_and_metadata(
        positions_1d=positions_1d,
        block_table_tensor=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        max_model_len=max_model_len,
        out_clamped_positions=out_clamped_positions,
        out_slot_mapping=out_slot_mapping,
    )


@pytest.mark.parametrize("num_speculative_tokens", [2, 4])
@pytest.mark.parametrize("has_rejected", [False, True])
def test_warmup_shapes_copy_expand_dflash(
    num_speculative_tokens: int, has_rejected: bool
) -> None:
    """The DFlash fused kernel must accept the shapes
    ``DFlashProposer.dry_run_helper_kernels`` builds, including the
    ``HAS_NUM_REJECTED`` constexpr toggled either way."""
    num_reqs = 1
    block_size = 16
    max_model_len = 4096
    num_query_per_req = 1 + num_speculative_tokens
    # Match production sizing: ``max_ctx_per_req == 1`` for the first request
    # after server start (no prior context).
    num_context = 1
    max_tokens_per_req = num_context + num_query_per_req
    block_size_kernel = min(256, next_power_of_2(max_tokens_per_req))
    num_blocks_grid = (max_tokens_per_req + block_size_kernel - 1) // block_size_kernel
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size

    next_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)
    target_positions = torch.zeros(num_context, dtype=torch.int64, device=DEVICE)
    out_input_ids = torch.zeros(num_query_per_req, dtype=torch.int32, device=DEVICE)
    out_context_positions = torch.zeros(num_context, dtype=torch.int64, device=DEVICE)
    out_query_positions = torch.zeros(
        num_query_per_req, dtype=torch.int64, device=DEVICE
    )
    out_context_slot = torch.zeros(num_context, dtype=torch.int64, device=DEVICE)
    out_query_slot = torch.zeros(num_query_per_req, dtype=torch.int64, device=DEVICE)
    out_token_indices = torch.empty(
        num_reqs * num_speculative_tokens, dtype=torch.int32, device=DEVICE
    )
    block_table = torch.zeros(
        (num_reqs, n_blocks_per_req), dtype=torch.int32, device=DEVICE
    )
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    qsl[1] = num_context
    num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE)

    copy_and_expand_dflash_inputs_kernel[(num_reqs, num_blocks_grid)](
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        out_input_ids_ptr=out_input_ids,
        out_context_positions_ptr=out_context_positions,
        out_query_positions_ptr=out_query_positions,
        out_context_slot_mapping_ptr=out_context_slot,
        out_query_slot_mapping_ptr=out_query_slot,
        out_token_indices_ptr=out_token_indices,
        block_table_ptr=block_table,
        block_table_stride=block_table.stride(0),
        query_start_loc_ptr=qsl,
        num_rejected_tokens_ptr=num_rejected if has_rejected else 0,
        parallel_drafting_token_id=0,
        block_size=block_size,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=num_speculative_tokens,
        total_input_tokens=num_context,
        BLOCK_SIZE=block_size_kernel,
        HAS_NUM_REJECTED=has_rejected,
    )
    torch.accelerator.synchronize()


def test_warmup_shapes_update_num_computed_tokens() -> None:
    """The async-spec-decode ``@torch.compile`` correction must accept the
    shapes ``Worker._warmup_spec_decode_helpers`` builds (V1 path)."""
    max_num_reqs = 8
    num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=DEVICE)
    num_accepted_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=DEVICE)
    prev_positions = torch.zeros(max_num_reqs, dtype=torch.int64, device=DEVICE)
    valid_sampled_token_count = torch.zeros(
        max_num_reqs, dtype=torch.int32, device=DEVICE
    )
    prev_num_draft_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=DEVICE)
    cpu_num_computed_tokens = torch.zeros(
        max_num_reqs, dtype=torch.int32, device=DEVICE
    )
    update_num_computed_tokens_for_batch_change(
        num_computed_tokens,
        num_accepted_tokens,
        prev_positions,
        valid_sampled_token_count,
        prev_num_draft_tokens,
        cpu_num_computed_tokens,
    )
    torch.accelerator.synchronize()
