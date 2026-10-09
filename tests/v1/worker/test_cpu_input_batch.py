# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.cpu.input_batch import get_num_sampled_and_rejected


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
