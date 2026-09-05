# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU C++ replacements for the spec-decode Triton kernels.

``CPUModelRunner._postprocess_triton`` swaps these in when Triton-CPU is
unavailable, so they are the kernels CPU speculative decoding actually runs.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("CPU spec-decode kernels", allow_module_level=True)

import vllm._custom_ops  # noqa: F401
import vllm.v1.spec_decode.utils as spec_decode_utils
from vllm.utils import cpu_triton_utils as cpu_tl
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    eagle_step_update_slot_mapping_and_metadata,
)

# cpu_triton_utils dispatches straight to torch.ops._C, which only exists once
# the CPU extension is loaded by the _custom_ops import above.
if not hasattr(torch.ops._C, "eagle_step_slot_mapping_metadata_kernel_impl"):
    pytest.skip("vLLM built without the CPU extension", allow_module_level=True)


def _reference_eagle_step_slot_mapping(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_positions = positions_1d + 1
    exceeds_max = new_positions >= max_model_len
    clamped_positions = torch.where(
        exceeds_max, torch.zeros_like(positions_1d), new_positions
    )
    block_numbers = (clamped_positions // block_size).clamp(
        max=block_table_tensor.shape[1] - 1
    )
    block_ids = block_table_tensor[
        torch.arange(positions_1d.shape[0]), block_numbers.long()
    ].long()
    slot_mapping = block_ids * block_size + (clamped_positions % block_size)
    slot_mapping = torch.where(
        exceeds_max, torch.full_like(slot_mapping, PADDING_SLOT_ID), slot_mapping
    )
    new_seq_lens = torch.where(exceeds_max, torch.ones_like(seq_lens), seq_lens + 1)
    return clamped_positions, slot_mapping, new_seq_lens.clamp(max=max_model_len)


@pytest.fixture
def cpu_eagle_step_kernel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        spec_decode_utils,
        "eagle_step_slot_mapping_metadata_kernel",
        cpu_tl.eagle_step_slot_mapping_metadata_kernel,
    )


@pytest.mark.parametrize("input_batch_size", [8, 12])
def test_eagle_step_slot_mapping(cpu_eagle_step_kernel, input_batch_size):
    batch_size = 8
    block_size = 16
    max_model_len = 256
    n_blocks_per_req = max_model_len // block_size

    torch.manual_seed(0)
    positions = torch.randint(0, max_model_len - 10, (batch_size,), dtype=torch.int64)
    # Exercise the clamp-to-PADDING_SLOT_ID branch.
    positions[-1] = max_model_len - 1
    block_table = torch.randint(
        0, 100, (batch_size, n_blocks_per_req), dtype=torch.int32
    )
    seq_lens = torch.randint(1, 64, (batch_size,), dtype=torch.int32)

    ref_clamped, ref_slot, ref_seq_lens = _reference_eagle_step_slot_mapping(
        positions.clone(), block_table, seq_lens.clone(), block_size, max_model_len
    )

    out_clamped = torch.zeros(batch_size, dtype=torch.int64)
    out_slot = torch.full((input_batch_size,), -999, dtype=torch.int64)
    out_seq_lens = seq_lens.clone()
    eagle_step_update_slot_mapping_and_metadata(
        positions_1d=positions,
        block_table_tensor=block_table,
        seq_lens=out_seq_lens,
        block_size=block_size,
        max_model_len=max_model_len,
        out_clamped_positions=out_clamped,
        out_slot_mapping=out_slot,
        input_batch_size=input_batch_size,
    )

    assert torch.equal(out_clamped, ref_clamped)
    assert torch.equal(out_slot[:batch_size], ref_slot)
    assert torch.equal(out_seq_lens, ref_seq_lens)
    assert (out_slot[batch_size:] == PADDING_SLOT_ID).all()


def test_eagle_prepare_inputs_padded():
    num_draft_per_req = [3, 0, 2]
    valid_counts = [2, 1, 3]
    num_reqs = len(num_draft_per_req)

    cu_num_draft = torch.tensor(num_draft_per_req, dtype=torch.int32).cumsum(0).int()
    valid_sampled = torch.tensor(valid_counts, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4, 6, 11], dtype=torch.int32)

    token_indices = torch.zeros(num_reqs, dtype=torch.int32)
    num_rejected = torch.zeros(num_reqs, dtype=torch.int32)
    cpu_tl.eagle_prepare_inputs_padded_kernel[(num_reqs,)](
        cu_num_draft,
        valid_sampled,
        query_start_loc,
        token_indices,
        num_rejected,
        num_reqs,
    )

    for req in range(num_reqs):
        num_draft = num_draft_per_req[req]
        expected_rejected = num_draft + 1 - valid_counts[req] if num_draft else 0
        expected_index = query_start_loc[req + 1].item() - 1 - expected_rejected
        assert num_rejected[req].item() == expected_rejected
        assert token_indices[req].item() == expected_index


def test_eagle_prepare_next_token_padded():
    vocab_size = 100
    num_sampled_per_req = 3
    # Rejected slots are -1; out-of-vocab ids are also treated as invalid.
    sampled = torch.tensor(
        [
            [5, 7, -1],
            [-1, -1, -1],
            [1, vocab_size + 4, -1],
            [9, 9, 9],
        ],
        dtype=torch.int64,
    )
    discard = torch.tensor([False, False, False, True])
    backup = torch.tensor([42, 43, 44, 45], dtype=torch.int64)
    num_reqs = sampled.shape[0]

    next_token_ids = torch.zeros(num_reqs, dtype=torch.int64)
    valid_counts = torch.zeros(num_reqs, dtype=torch.int32)
    cpu_tl.eagle_prepare_next_token_padded_kernel[(num_reqs,)](
        sampled,
        discard,
        backup,
        next_token_ids,
        valid_counts,
        vocab_size,
        num_sampled_per_req,
        num_reqs,
    )

    assert valid_counts.tolist() == [2, 0, 1, 0]
    assert next_token_ids.tolist() == [7, 43, 1, 45]


def test_rejection_greedy_sample():
    num_draft_per_req = [3, 2, 2]
    max_spec_len = 3
    batch_size = len(num_draft_per_req)

    cu_num_draft = torch.tensor(num_draft_per_req, dtype=torch.int32).cumsum(0).int()
    # Request 0 accepts everything, 1 diverges at its second token, 2 is
    # non-greedy and must be left untouched.
    draft = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    target_argmax = torch.tensor([1, 2, 3, 4, 55, 6, 7], dtype=torch.int64)
    bonus = torch.tensor([[11], [12], [13]], dtype=torch.int64)
    is_greedy = torch.tensor([True, True, False])

    output = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64)
    cpu_tl.rejection_greedy_sample_kernel[(batch_size,)](
        output,
        cu_num_draft,
        draft,
        target_argmax,
        bonus,
        is_greedy,
        max_spec_len,
    )

    assert output[0].tolist() == [1, 2, 3, 11]
    assert output[1].tolist() == [4, 55, -1, -1]
    assert output[2].tolist() == [-1, -1, -1, -1]


def test_copy_and_expand_dflash_inputs():
    # The wrapper silently falls back to a Python loop when the op is absent,
    # which would make this test pass without exercising the kernel.
    assert hasattr(torch.ops._C, "copy_and_expand_dflash_inputs_kernel_impl")

    num_speculative_tokens = 2
    num_query_per_req = num_speculative_tokens + 1
    block_size = 4
    n_blocks_per_req = 8
    parallel_drafting_token_id = 0

    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
    num_reqs = query_start_loc.shape[0] - 1
    total_input_tokens = int(query_start_loc[-1])
    target_positions = torch.tensor([0, 1, 2, 10, 11, 12, 13], dtype=torch.int64)
    next_token_ids = torch.tensor([21, 22], dtype=torch.int64)
    block_table = torch.arange(num_reqs * n_blocks_per_req, dtype=torch.int32).reshape(
        num_reqs, n_blocks_per_req
    )
    num_rejected = torch.tensor([0, 1], dtype=torch.int32)

    out_input_ids = torch.zeros(num_reqs * num_query_per_req, dtype=torch.int64)
    out_ctx_positions = torch.zeros(total_input_tokens, dtype=torch.int64)
    out_query_positions = torch.zeros(num_reqs * num_query_per_req, dtype=torch.int64)
    out_ctx_slots = torch.zeros(total_input_tokens, dtype=torch.int64)
    out_query_slots = torch.zeros(num_reqs * num_query_per_req, dtype=torch.int64)
    out_token_indices = torch.zeros(
        num_reqs * num_speculative_tokens, dtype=torch.int32
    )

    cpu_tl.copy_and_expand_dflash_inputs_kernel[(num_reqs, 1)](
        next_token_ids,
        target_positions,
        out_input_ids,
        out_ctx_positions,
        out_query_positions,
        out_ctx_slots,
        out_query_slots,
        out_token_indices,
        block_table,
        block_table.stride(0),
        query_start_loc,
        num_rejected,
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_tokens,
        total_input_tokens,
        HAS_NUM_REJECTED=True,
    )

    def slot(req: int, position: int) -> int:
        block_num = min(position // block_size, block_table.stride(0) - 1)
        return int(block_table[req, block_num]) * block_size + position % block_size

    # Context positions are copied through unchanged.
    assert torch.equal(out_ctx_positions, target_positions)
    expected_ctx_slots = [
        slot(req, int(target_positions[i]))
        for req in range(num_reqs)
        for i in range(int(query_start_loc[req]), int(query_start_loc[req + 1]))
    ]
    assert out_ctx_slots.tolist() == expected_ctx_slots

    # Queries continue from the last accepted context position, so request 1
    # starts at 13 rather than 14 because one of its tokens was rejected.
    expected_query_positions = [3, 4, 5, 13, 14, 15]
    assert out_query_positions.tolist() == expected_query_positions
    assert out_query_slots.tolist() == [
        slot(i // num_query_per_req, p) for i, p in enumerate(expected_query_positions)
    ]

    # First query slot carries the sampled token, the rest are drafting masks.
    assert out_input_ids.tolist() == [
        21,
        parallel_drafting_token_id,
        parallel_drafting_token_id,
        22,
        parallel_drafting_token_id,
        parallel_drafting_token_id,
    ]
    assert out_token_indices.tolist() == [1, 2, 4, 5]
