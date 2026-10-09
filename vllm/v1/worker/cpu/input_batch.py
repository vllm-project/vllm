# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU implementations for Model Runner V2 input-batch kernels."""

import torch


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    is_chunked_prefilling = seq_lens[:num_reqs] < prefill_len[idx_mapping]
    num_sampled.masked_fill_(is_chunked_prefilling, 0)
    num_rejected.copy_(cu_num_logits[1 : num_reqs + 1] - cu_num_logits[:num_reqs])
    num_rejected.sub_(num_sampled)
    num_rejected.masked_fill_(is_chunked_prefilling, 0)
    return num_sampled, num_rejected
