# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DFlash first-pass slot mapping."""

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.platforms import current_platform
from vllm.v1.worker.block_table import BlockTable

DEVICE_TYPE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="CUDA/XPU required for DFlash kernel tests",
)


def test_dflash_first_prefill_query_slots_are_request_owned():
    """DFlash first-pass query slots must address allocated request blocks.

    This test links the scheduler output to the real DFlash input expansion
    kernel. The kernel generates query positions immediately after the first
    prefill context; those positions must map to logical blocks that the
    scheduler already allocated for the request.
    """
    pytest.importorskip("triton")
    from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel

    device = torch.device(DEVICE_TYPE)
    block_size = 16
    num_speculative_tokens = 3
    num_query_per_req = 1 + num_speculative_tokens
    num_context_tokens = block_size
    max_blocks_per_req = 2

    scheduler = create_scheduler(
        block_size=block_size,
        max_num_batched_tokens=64,
    )
    scheduler.use_dflash = True
    scheduler.num_lookahead_tokens = num_speculative_tokens

    (request,) = create_requests(
        num_requests=1,
        num_tokens=num_context_tokens,
        block_size=block_size,
    )
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    assert scheduler_output.num_scheduled_tokens[request.request_id] == block_size

    block_table = BlockTable(
        block_size=block_size,
        max_num_reqs=1,
        max_num_blocks_per_req=max(max_blocks_per_req, len(block_ids)),
        max_num_batched_tokens=num_context_tokens + num_query_per_req,
        pin_memory=False,
        device=device,
        kernel_block_size=block_size,
        cp_kv_cache_interleave_size=1,
    )
    block_table.add_row(block_ids, row_idx=0)
    block_table.commit_block_table(num_reqs=1)
    block_table_tensor = block_table.get_device_tensor(num_reqs=1)

    next_token_ids = torch.tensor([123], dtype=torch.int32, device=device)
    target_positions = torch.arange(
        num_context_tokens, dtype=torch.int64, device=device
    )
    query_start_loc = torch.tensor(
        [0, num_context_tokens], dtype=torch.int32, device=device
    )

    out_input_ids = torch.empty(num_query_per_req, dtype=torch.int32, device=device)
    out_context_positions = torch.empty(
        num_context_tokens, dtype=torch.int64, device=device
    )
    out_query_positions = torch.empty(
        num_query_per_req, dtype=torch.int64, device=device
    )
    out_context_slot_mapping = torch.empty(
        num_context_tokens, dtype=torch.int64, device=device
    )
    out_query_slot_mapping = torch.empty(
        num_query_per_req, dtype=torch.int64, device=device
    )
    out_token_indices = torch.empty(
        num_speculative_tokens, dtype=torch.int32, device=device
    )

    copy_and_expand_dflash_inputs_kernel[(1, 1)](
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        out_input_ids_ptr=out_input_ids,
        out_context_positions_ptr=out_context_positions,
        out_query_positions_ptr=out_query_positions,
        out_context_slot_mapping_ptr=out_context_slot_mapping,
        out_query_slot_mapping_ptr=out_query_slot_mapping,
        out_token_indices_ptr=out_token_indices,
        block_table_ptr=block_table_tensor,
        block_table_stride=block_table_tensor.stride(0),
        query_start_loc_ptr=query_start_loc,
        num_rejected_tokens_ptr=0,
        parallel_drafting_token_id=42,
        block_size=block_size,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=num_speculative_tokens,
        total_input_tokens=num_context_tokens,
        BLOCK_SIZE=32,
        HAS_NUM_REJECTED=False,
    )

    expected_query_positions = torch.arange(
        block_size,
        block_size + num_query_per_req,
        dtype=torch.int64,
        device=device,
    )
    assert torch.equal(out_query_positions, expected_query_positions)

    query_logical_blocks = out_query_positions // block_size
    assert torch.all(query_logical_blocks < len(block_ids)), (
        "DFlash generated query positions that address logical blocks "
        f"{query_logical_blocks.cpu().tolist()}, but the scheduler only "
        f"allocated {len(block_ids)} request blocks: {block_ids}. "
        f"Kernel slot mapping was {out_query_slot_mapping.cpu().tolist()}."
    )

    mapped_physical_blocks = (out_query_slot_mapping // block_size).cpu().tolist()
    assert all(block_id in block_ids for block_id in mapped_physical_blocks), (
        "DFlash query slots mapped to physical blocks outside the request-owned "
        f"block ids. mapped={mapped_physical_blocks}, owned={block_ids}."
    )
