# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.cpu.input_batch import (
    get_num_sampled_and_rejected,
    prepare_prefill_inputs,
)


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

    idx_mapping = torch.tensor([4, 1, 3], dtype=torch.int32)
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


def test_get_num_sampled_and_rejected_cpu() -> None:
    num_sampled = torch.tensor([1, 2, 2, 1], dtype=torch.int32)
    seq_lens = torch.tensor([4, 6, 22, 7], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 1, 5, 8, 10], dtype=torch.int32)
    idx_mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
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
