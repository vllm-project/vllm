# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.worker.cpu.input_batch import (
    combine_sampled_and_draft_tokens,
    get_num_sampled_and_rejected,
    post_update,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.cpu.model_state import CPURopeState

pytestmark = pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")


def test_prepare_prefill_inputs_cpu() -> None:
    input_ids_storage = torch.full((8,), -1, dtype=torch.int32)
    input_ids = input_ids_storage[1:7]
    next_prefill_storage = torch.full((3, 8), -1, dtype=torch.int32)
    next_prefill_tokens = next_prefill_storage[:, 1:7]
    all_token_ids_storage = torch.full((6, 12), -1, dtype=torch.int32)
    all_token_ids = all_token_ids_storage[:, 1:11]
    for req_state_idx in range(6):
        all_token_ids[req_state_idx] = req_state_idx * 100 + torch.arange(
            10, dtype=torch.int32
        )

    idx_mapping = torch.tensor([4, 1, 3], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
    prefill_len = torch.tensor([0, 4, 0, 8, 7, 0], dtype=torch.int32)
    num_computed_tokens = torch.tensor([0, 4, 0, 4, 2, 0], dtype=torch.int32)

    prepare_prefill_inputs(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )

    assert input_ids.tolist() == [402, 403, 404, -1, -1, 304]
    assert next_prefill_tokens[:, 4].tolist() == [405, 406, 0]
    assert next_prefill_tokens[:, 3].tolist() == [305, 306, 307]
    assert next_prefill_tokens[:, 1].tolist() == [-1, -1, -1]
    assert torch.all(next_prefill_tokens[:, [0, 2, 5]] == -1)


def test_get_prompt_logprobs_token_ids_cpu() -> None:
    all_token_ids_storage = torch.full((6, 12), -1, dtype=torch.int32)
    all_token_ids = all_token_ids_storage[:, 1:11]
    for req_state_idx in range(6):
        all_token_ids[req_state_idx] = req_state_idx * 100 + torch.arange(
            10, dtype=torch.int32
        )

    token_ids = ops.get_prompt_logprobs_token_ids_cpu(
        num_tokens=6,
        query_start_loc=torch.tensor([0, 3, 5, 6], dtype=torch.int32),
        idx_mapping=torch.tensor([4, 1, 3], dtype=torch.int64),
        num_computed_tokens=torch.tensor([0, 4, 0, 1, 2, 0], dtype=torch.int32),
        all_token_ids=all_token_ids,
    )

    assert token_ids.dtype == torch.int64
    assert token_ids.tolist() == [403, 404, 405, 105, 106, 302]


def test_prepare_pos_seq_lens_cpu() -> None:
    positions_storage = torch.full((8,), -1, dtype=torch.int64)
    positions = positions_storage[1:]
    seq_lens_storage = torch.full((8,), -1, dtype=torch.int32)
    seq_lens = seq_lens_storage[1:7]

    idx_mapping = torch.tensor([3, 1, 4], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
    num_computed_tokens = torch.tensor([0, 7, 0, 2, 10], dtype=torch.int32)

    prepare_pos_seq_lens(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        positions,
        seq_lens,
    )

    assert positions.tolist() == [2, 3, 4, 7, 8, 10, -1]
    assert seq_lens.tolist() == [5, 9, 11, 0, 0, 0]
    assert positions_storage[0].item() == -1
    assert seq_lens_storage[[0, 7]].tolist() == [-1, -1]


def test_prepare_mrope_positions_cpu() -> None:
    num_dims = 3
    positions_storage = torch.full((num_dims, 10), -1, dtype=torch.int64)
    positions = positions_storage[:, 1:9]

    prefill_positions_storage = torch.full((15, 10), -1, dtype=torch.int32)
    prefill_positions = prefill_positions_storage[:, 1:9]
    for req_state_idx in range(5):
        for dim in range(num_dims):
            prefill_positions[req_state_idx * num_dims + dim] = (
                req_state_idx * 1000 + dim * 100 + torch.arange(8, dtype=torch.int32)
            )

    ops.prepare_mrope_positions_cpu(
        positions,
        prefill_positions,
        prefill_delta=torch.tensor([0, 10, 0, 20, 30], dtype=torch.int32),
        idx_mapping=torch.tensor([3, 1, 4], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 3, 6, 8], dtype=torch.int32),
        prefill_lens=torch.tensor([0, 5, 0, 6, 4], dtype=torch.int32),
        num_computed_tokens=torch.tensor([0, 5, 0, 2, 3], dtype=torch.int32),
    )

    assert positions.tolist() == [
        [3002, 3003, 3004, 15, 16, 17, 4003, 34],
        [3102, 3103, 3104, 15, 16, 17, 4103, 34],
        [3202, 3203, 3204, 15, 16, 17, 4203, 34],
    ]
    assert torch.all(positions_storage[:, [0, 9]] == -1)


def test_cpu_rope_state_prepares_positions() -> None:
    state = object.__new__(CPURopeState)
    state.positions = torch.full((2, 2), -1, dtype=torch.int64)
    state.prefill_positions = SimpleNamespace(
        gpu=torch.tensor([[-1, -1], [-1, -1], [100, -1], [200, -1]], dtype=torch.int32)
    )
    state.prefill_delta = SimpleNamespace(gpu=torch.tensor([0, 10], dtype=torch.int32))

    state.prepare_positions(
        idx_mapping=torch.tensor([1], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        prefill_lens=torch.tensor([0, 1], dtype=torch.int32),
        num_computed_tokens=torch.tensor([0, 0], dtype=torch.int32),
    )

    assert state.positions.tolist() == [[100, 11], [200, 11]]


def test_combine_sampled_and_draft_tokens_cpu_basic() -> None:
    input_ids = torch.tensor([10, 11, 12, -1], dtype=torch.int32)
    idx_mapping = torch.tensor([3, 1], dtype=torch.int64)
    last_sampled_tokens = torch.tensor([[40], [41], [42], [43]])
    query_start_loc = torch.tensor([0, 3, 4], dtype=torch.int32)
    seq_lens = torch.tensor([5, 6], dtype=torch.int32)
    prefill_len = torch.tensor([0, 5, 0, 5], dtype=torch.int32)
    draft_tokens = torch.empty((4, 0), dtype=torch.int64)
    cu_num_logits = torch.tensor([0, 1, 2], dtype=torch.int32)

    logits_indices = combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits=2,
    )

    assert input_ids.tolist() == [10, 11, 12, 41]
    assert logits_indices.tolist() == [2, 3]


def test_combine_sampled_and_draft_tokens_cpu_with_drafts() -> None:
    input_ids_storage = torch.arange(99, 108, dtype=torch.int32)
    input_ids = input_ids_storage[1:8]
    idx_mapping = torch.tensor([3, 0], dtype=torch.int64)
    last_sampled_tokens = torch.tensor([[50], [150], [250], [350]])
    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
    seq_lens = torch.tensor([10, 6], dtype=torch.int32)
    prefill_len = torch.tensor([5, 0, 0, 5], dtype=torch.int32)
    draft_tokens_storage = torch.full((4, 4), -1, dtype=torch.int64)
    draft_tokens = draft_tokens_storage[:, 1:3]
    draft_tokens.copy_(torch.tensor([[11, 12], [111, 112], [211, 212], [311, 312]]))
    cu_num_logits = torch.tensor([0, 3, 6], dtype=torch.int32)

    logits_indices = combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits=6,
    )

    assert input_ids.tolist() == [350, 311, 312, 103, 104, 11, 12]
    assert logits_indices.tolist() == [0, 1, 2, 4, 5, 6]
    assert input_ids_storage[[0, 8]].tolist() == [99, 107]


def test_get_num_sampled_and_rejected_cpu() -> None:
    num_sampled = torch.tensor([1, 2, 2, 1], dtype=torch.int32)
    seq_lens = torch.tensor([4, 6, 22, 7], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 1, 5, 8, 10], dtype=torch.int32)
    idx_mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    prefill_len = torch.tensor([10, 7, 4, 20], dtype=torch.int32)

    actual_sampled, actual_rejected = get_num_sampled_and_rejected(
        num_sampled,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )

    assert torch.equal(actual_sampled, torch.tensor([1, 0, 2, 1]))
    assert torch.equal(actual_rejected, torch.tensor([0, 0, 1, 1]))


def test_post_update_cpu() -> None:
    idx_mapping = torch.tensor([3, -1, 1, 4], dtype=torch.int64)
    num_computed_tokens = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
    last_sampled_storage = torch.full((5, 2), -1, dtype=torch.int64)
    last_sampled_tokens = last_sampled_storage[:, 1:]
    output_bin_counts_storage = torch.zeros((5, 10), dtype=torch.int32)
    output_bin_counts = output_bin_counts_storage[:, 1:9]
    sampled_tokens_storage = torch.full((4, 5), -1, dtype=torch.int64)
    sampled_tokens = sampled_tokens_storage[:, 1:4]
    sampled_tokens.copy_(torch.tensor([[2, 2, 5], [7, 6, 5], [4, 1, 0], [6, 3, 0]]))
    num_sampled = torch.tensor([3, 1, 0, 2], dtype=torch.int32)
    num_rejected = torch.tensor([1, 0, 2, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4, 5, 8, 10], dtype=torch.int32)
    all_token_ids_storage = torch.full((5, 10), -1, dtype=torch.int32)
    all_token_ids = all_token_ids_storage[:, 1:9]
    total_len = torch.tensor([1, 3, 1, 2, 4], dtype=torch.int32)

    post_update(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )

    assert num_computed_tokens.tolist() == [10, 21, 30, 43, 51]
    assert last_sampled_tokens.squeeze(1).tolist() == [-1, -1, -1, 5, 3]
    assert total_len.tolist() == [1, 3, 1, 5, 6]
    assert all_token_ids[3].tolist() == [-1, -1, 2, 2, 5, -1, -1, -1]
    assert all_token_ids[4].tolist() == [-1, -1, -1, -1, 6, 3, -1, -1]
    assert torch.all(all_token_ids[[0, 1, 2]] == -1)
    assert output_bin_counts[3].tolist() == [0, 0, 2, 0, 0, 1, 0, 0]
    assert output_bin_counts[4].tolist() == [0, 0, 0, 1, 0, 0, 1, 0]
    assert torch.all(output_bin_counts[[0, 1, 2]] == 0)


def test_post_update_cpu_without_optional_inputs() -> None:
    num_computed_tokens = torch.tensor([8, 13], dtype=torch.int32)
    last_sampled_tokens = torch.tensor([[70], [80]], dtype=torch.int64)
    all_token_ids = torch.full((2, 6), -1, dtype=torch.int32)
    total_len = torch.tensor([2, 3], dtype=torch.int32)

    post_update(
        idx_mapping=torch.tensor([1], dtype=torch.int64),
        num_computed_tokens=num_computed_tokens,
        last_sampled_tokens=last_sampled_tokens,
        output_bin_counts=None,
        sampled_tokens=torch.tensor([[4, 5]], dtype=torch.int64),
        num_sampled=torch.tensor([1], dtype=torch.int32),
        num_rejected=torch.tensor([2], dtype=torch.int32),
        query_start_loc=None,
        all_token_ids=all_token_ids,
        total_len=total_len,
    )

    assert num_computed_tokens.tolist() == [8, 11]
    assert last_sampled_tokens.squeeze(1).tolist() == [70, 4]
    assert total_len.tolist() == [2, 4]
    assert all_token_ids[1].tolist() == [-1, -1, -1, 4, -1, -1]


def test_post_update_num_computed_tokens_cpu() -> None:
    num_computed_tokens = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)

    post_update_num_computed_tokens(
        idx_mapping=torch.tensor([3, 1, 4], dtype=torch.int64),
        num_computed_tokens=num_computed_tokens,
        query_start_loc=torch.tensor([0, 3, 5, 6], dtype=torch.int32),
    )

    assert num_computed_tokens.tolist() == [10, 22, 30, 43, 51]
