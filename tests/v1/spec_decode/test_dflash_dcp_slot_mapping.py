# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP-aware slot mapping in the DFlash/DSpark input-prep kernel.

Guards against the dcp-blind draft slot mapping (global ``pos // block_size``
on a DCP-reduced block table) that collapsed acceptance to ~0% under
decode_context_parallel_size > 1.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

pytest.importorskip("triton")
if not current_platform.is_cuda_alike():
    pytest.skip("CUDA required for DFlash kernel tests", allow_module_level=True)

DEVICE = "cuda"
BLOCK_SIZE = 16
NUM_SPECULATIVE_STEPS = 3
NUM_QUERY_PER_REQ = 1 + NUM_SPECULATIVE_STEPS
MAX_NUM_REQS = 8
MAX_NUM_TOKENS = 512
MAX_NUM_BLOCKS = 64
MAX_MODEL_LEN = 4096


def _ref_slots(
    positions: torch.Tensor,
    block_row: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    cp_interleave: int,
) -> torch.Tensor:
    """Reference for the interleaved DCP slot layout (one block-table entry
    covers BLOCK_SIZE * cp_size global positions)."""
    virtual_block = BLOCK_SIZE * cp_size
    block_indices = (positions // virtual_block).clamp(max=block_row.shape[0] - 1)
    block_offsets = positions % virtual_block
    block_ids = block_row[block_indices].long()
    if cp_size == 1:
        return block_ids * BLOCK_SIZE + block_offsets
    is_local = (block_offsets // cp_interleave) % cp_size == cp_rank
    local_offsets = (
        block_offsets // (cp_interleave * cp_size) * cp_interleave
        + block_offsets % cp_interleave
    )
    slots = block_ids * BLOCK_SIZE + local_offsets
    return torch.where(is_local, slots, torch.full_like(slots, PAD_SLOT_ID))


def _run_prepare(cp_size: int, cp_rank: int, cp_interleave: int):
    # Three requests mimicking a mixed batch: decode with rejects,
    # single-token decode, and a chunked-prefill tail.
    ctx_positions = [
        torch.arange(100, 120, device=DEVICE),
        torch.arange(0, 1, device=DEVICE),
        torch.arange(37, 100, device=DEVICE),
    ]
    num_rejected = torch.tensor([2, 0, 0], dtype=torch.int32, device=DEVICE)
    num_sampled = torch.tensor([1, 1, 0], dtype=torch.int32, device=DEVICE)
    num_reqs = len(ctx_positions)

    positions = torch.cat(ctx_positions)
    num_ctx = torch.tensor([len(p) for p in ctx_positions], device=DEVICE)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    query_start_loc[1:] = num_ctx.cumsum(0)
    num_tokens = int(query_start_loc[-1])

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=num_ctx.cpu().numpy(),
        positions=positions,
        query_start_loc=query_start_loc,
        idx_mapping=torch.arange(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE),
    )
    input_buffers = InputBuffers(MAX_NUM_REQS, MAX_NUM_TOKENS, torch.device(DEVICE))

    generator = torch.Generator(device=DEVICE).manual_seed(42)
    block_table = (
        torch.randperm(
            MAX_NUM_REQS * MAX_NUM_BLOCKS, generator=generator, device=DEVICE
        )
        .view(MAX_NUM_REQS, MAX_NUM_BLOCKS)
        .to(torch.int32)
    )

    query_slot_mapping = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE)
    context_positions = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE)
    context_slot_mapping = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE)
    max_num_sampled = MAX_NUM_REQS * NUM_SPECULATIVE_STEPS
    sample_indices = torch.zeros(max_num_sampled, dtype=torch.int64, device=DEVICE)
    sample_pos = torch.zeros(max_num_sampled, dtype=torch.int64, device=DEVICE)
    sample_idx_mapping = torch.zeros(max_num_sampled, dtype=torch.int32, device=DEVICE)
    last_sampled = torch.full((MAX_NUM_REQS,), 7, dtype=torch.int64, device=DEVICE)
    next_prefill_tokens = torch.full(
        (MAX_NUM_REQS,), 11, dtype=torch.int64, device=DEVICE
    )

    prepare_dflash_inputs(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        block_table,
        BLOCK_SIZE,
        parallel_drafting_token_id=1,
        num_query_per_req=NUM_QUERY_PER_REQ,
        num_speculative_steps=NUM_SPECULATIVE_STEPS,
        max_num_reqs=MAX_NUM_REQS,
        max_num_tokens=MAX_NUM_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
    )
    return SimpleNamespace(
        ctx_positions=ctx_positions,
        num_rejected=num_rejected,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        query_start_loc=query_start_loc,
        block_table=block_table,
        input_buffers=input_buffers,
        query_slot_mapping=query_slot_mapping,
        context_positions=context_positions,
        context_slot_mapping=context_slot_mapping,
    )


@pytest.mark.parametrize(
    "cp_size,cp_interleave", [(1, 1), (2, 1), (4, 1), (2, 16), (4, 8)]
)
def test_dflash_slot_mapping_matches_dcp_layout(cp_size: int, cp_interleave: int):
    for cp_rank in range(cp_size):
        r = _run_prepare(cp_size, cp_rank, cp_interleave)

        for req in range(r.num_reqs):
            start = int(r.query_start_loc[req])
            end = int(r.query_start_loc[req + 1])
            ctx_pos = r.ctx_positions[req]
            ref_ctx = _ref_slots(
                ctx_pos, r.block_table[req], cp_size, cp_rank, cp_interleave
            )
            torch.testing.assert_close(
                r.context_slot_mapping[start:end], ref_ctx, rtol=0, atol=0
            )
            torch.testing.assert_close(
                r.context_positions[start:end], ctx_pos, rtol=0, atol=0
            )

            last_valid_pos = int(ctx_pos[-1 - int(r.num_rejected[req])])
            query_pos = torch.arange(
                last_valid_pos + 1,
                last_valid_pos + 1 + NUM_QUERY_PER_REQ,
                device=DEVICE,
            )
            # Under DCP the dense flash path attends the query block against
            # freshly computed K/V (never the cache), so the query K/V is not
            # written: every query slot is PAD. Without DCP it is the global
            # slot.
            if cp_size == 1:
                ref_q = _ref_slots(
                    query_pos, r.block_table[req], cp_size, cp_rank, cp_interleave
                )
            else:
                ref_q = torch.full_like(query_pos, PAD_SLOT_ID)
            qstart = req * NUM_QUERY_PER_REQ
            torch.testing.assert_close(
                r.query_slot_mapping[qstart : qstart + NUM_QUERY_PER_REQ],
                ref_q,
                rtol=0,
                atol=0,
            )

            # seq_lens and positions must stay GLOBAL under DCP: the dense
            # attention metadata builder localizes context lens itself, and
            # RoPE/Gumbel need global positions.
            assert (
                int(r.input_buffers.seq_lens[req])
                == last_valid_pos + 1 + NUM_QUERY_PER_REQ
            )
            torch.testing.assert_close(
                r.input_buffers.positions[qstart : qstart + NUM_QUERY_PER_REQ],
                query_pos.clamp(max=MAX_MODEL_LEN - 1),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("cp_size,cp_interleave", [(2, 1), (4, 1), (4, 8)])
def test_dflash_query_slots_all_pad_under_dcp(cp_size: int, cp_interleave: int):
    """Under DCP every query slot is PAD on every rank.

    The dense flash DCP path attends the query block against freshly computed
    K/V, so writing the local num_kv_heads query K/V into the DCP-replicated
    cache would corrupt the head layout; the write must be skipped entirely.
    """
    for cp_rank in range(cp_size):
        r = _run_prepare(cp_size, cp_rank, cp_interleave)
        num_query_tokens = r.num_reqs * NUM_QUERY_PER_REQ
        assert (r.query_slot_mapping[:num_query_tokens] == PAD_SLOT_ID).all()


def test_dflash_query_slots_written_without_dcp():
    """Without DCP the query slots are real (global) cache slots, not PAD."""
    r = _run_prepare(cp_size=1, cp_rank=0, cp_interleave=1)
    num_query_tokens = r.num_reqs * NUM_QUERY_PER_REQ
    assert (r.query_slot_mapping[:num_query_tokens] != PAD_SLOT_ID).all()


@pytest.mark.parametrize("cp_size,cp_interleave", [(2, 1), (4, 1), (4, 8)])
def test_dflash_slot_mapping_cross_rank_partition(cp_size: int, cp_interleave: int):
    """Each context position is written by exactly one DCP rank, and owned
    slots are unique within a rank."""
    per_rank = [_run_prepare(cp_size, rank, cp_interleave) for rank in range(cp_size)]
    num_tokens = per_rank[0].num_tokens
    owned = torch.stack(
        [r.context_slot_mapping[:num_tokens] != PAD_SLOT_ID for r in per_rank]
    )
    assert (owned.int().sum(dim=0) == 1).all()
    for rank, r in enumerate(per_rank):
        slots = r.context_slot_mapping[:num_tokens][owned[rank]]
        assert slots.unique().numel() == slots.numel()
