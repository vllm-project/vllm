# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.utils import (
    update_num_computed_tokens_for_batch_change,
)


def test_update_num_computed_tokens_skips_stale_positive_prev_position():
    num_computed_tokens = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([9, 8, 7], dtype=torch.int32)
    prev_positions = torch.tensor([0, 2, -1], dtype=torch.int64)
    valid_sampled_token_count = torch.tensor([3, 4], dtype=torch.int32)
    prev_num_draft_tokens = torch.tensor([2, 0, 5, 0, 0, 0], dtype=torch.int32)
    cpu_num_computed_tokens = torch.tensor([100, 200, 300], dtype=torch.int32)

    update_num_computed_tokens_for_batch_change(
        num_computed_tokens,
        num_accepted_tokens,
        prev_positions,
        valid_sampled_token_count,
        prev_num_draft_tokens,
        cpu_num_computed_tokens,
    )

    assert num_computed_tokens.tolist() == [13, 200, 300, 40, 50, 60]
    assert num_accepted_tokens.tolist() == [3, 8, 7]


def test_update_num_computed_tokens_handles_empty_valid_counts():
    num_computed_tokens = torch.tensor([10, 20, 30], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([9, 8], dtype=torch.int32)
    prev_positions = torch.tensor([-1, 2], dtype=torch.int64)
    valid_sampled_token_count = torch.empty((0,), dtype=torch.int32)
    prev_num_draft_tokens = torch.tensor([2, 5, 7], dtype=torch.int32)
    cpu_num_computed_tokens = torch.tensor([100, 200], dtype=torch.int32)

    update_num_computed_tokens_for_batch_change(
        num_computed_tokens,
        num_accepted_tokens,
        prev_positions,
        valid_sampled_token_count,
        prev_num_draft_tokens,
        cpu_num_computed_tokens,
    )

    assert num_computed_tokens.tolist() == [100, 200, 30]
    assert num_accepted_tokens.tolist() == [9, 8]
