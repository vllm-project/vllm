# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        "DSpark sample layout test requires CUDA.",
        allow_module_level=True,
    )

from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel,
)


@pytest.mark.parametrize(
    (
        "sample_from_anchor",
        "sample_step_major",
        "expected_sample_indices",
    ),
    [
        (True, True, [0, 3, 6, 1, 4, 7, 2, 5, 8]),
        (False, True, [1, 5, 9, 2, 6, 10, 3, 7, 11]),
        (False, False, [1, 2, 3, 5, 6, 7, 9, 10, 11]),
    ],
)
def test_sample_indices_layout(
    sample_from_anchor: bool,
    sample_step_major: bool,
    expected_sample_indices: list[int],
):
    device = "cuda"
    num_reqs = 3
    num_speculative_steps = 3
    num_query_per_req = 3 if sample_from_anchor else 4
    max_num_reqs = 4
    max_num_tokens = max_num_reqs * num_query_per_req
    num_samples = max_num_reqs * num_speculative_steps

    out_input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
    out_query_positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
    out_query_start_loc = torch.zeros(
        max_num_reqs + 1, dtype=torch.int32, device=device
    )
    out_seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    out_query_slot_mapping = torch.zeros(
        max_num_tokens, dtype=torch.int64, device=device
    )
    out_context_positions = torch.zeros(num_reqs, dtype=torch.int64, device=device)
    out_context_slot_mapping = torch.zeros(num_reqs, dtype=torch.int64, device=device)
    out_sample_indices = torch.zeros(num_samples, dtype=torch.int64, device=device)
    out_sample_pos = torch.zeros(num_samples, dtype=torch.int64, device=device)
    out_sample_idx_mapping = torch.zeros(num_samples, dtype=torch.int32, device=device)
    out_temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
    out_seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)

    target_positions = torch.tensor([10, 20, 30], dtype=torch.int64, device=device)
    target_query_start_loc = torch.tensor(
        [0, 1, 2, 3], dtype=torch.int32, device=device
    )
    idx_mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device=device)
    last_sampled = torch.tensor([100, 101, 102, 0], dtype=torch.int32, device=device)
    next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=device)
    num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=device)
    seeds = torch.arange(max_num_reqs, dtype=torch.int64, device=device)
    block_table = torch.zeros((num_reqs, 16), dtype=torch.int32, device=device)

    _prepare_dflash_inputs_kernel[(num_reqs, 1)](
        out_input_ids,
        out_query_positions,
        out_query_start_loc,
        out_seq_lens,
        out_query_slot_mapping,
        out_context_positions,
        out_context_slot_mapping,
        out_sample_indices,
        out_sample_pos,
        out_sample_idx_mapping,
        out_temperature,
        out_seeds,
        target_positions,
        target_query_start_loc,
        idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        temperature,
        seeds,
        block_table,
        block_table.stride(0),
        999,
        16,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        1024,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        SAMPLE_STEP_MAJOR=sample_step_major,
        PAD_SLOT_ID=-1,
        BLOCK_SIZE=8,
    )

    active = num_reqs * num_speculative_steps
    if sample_step_major:
        sample_indices = out_sample_indices.view(num_speculative_steps, max_num_reqs)[
            :, :num_reqs
        ].reshape(active)
        padded_indices = out_sample_indices.view(num_speculative_steps, max_num_reqs)[
            :, num_reqs:
        ]
        assert padded_indices.eq(0).all()
    else:
        sample_indices = out_sample_indices[:active]
    assert sample_indices.cpu().tolist() == expected_sample_indices
    assert out_sample_pos[:active].cpu().tolist() == [
        12,
        13,
        14,
        22,
        23,
        24,
        32,
        33,
        34,
    ]
    assert out_sample_idx_mapping[:active].cpu().tolist() == [
        2,
        2,
        2,
        0,
        0,
        0,
        1,
        1,
        1,
    ]
