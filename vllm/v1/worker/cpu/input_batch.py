# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU implementations for Model Runner V2 input-batch kernels."""

import torch

from vllm import _custom_ops as ops


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    ops.prepare_prefill_inputs_cpu(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )


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
