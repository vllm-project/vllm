# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _logit_bias_kernel(
    logits_ptr,
    logits_stride,
    num_logit_bias_ptr,
    token_ids_ptr,
    token_ids_stride,
    bias_ptr,
    bias_stride,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    n = tl.load(num_logit_bias_ptr + req_idx)
    if n == 0:
        return

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < n
    token_ids = tl.load(token_ids_ptr + req_idx * token_ids_stride + block, mask=mask)
    bias = tl.load(bias_ptr + req_idx * bias_stride + block, mask=mask)

    logits = tl.load(logits_ptr + req_idx * logits_stride + token_ids, mask=mask)
    logits += bias
    tl.store(logits_ptr + req_idx * logits_stride + token_ids, logits, mask=mask)


def apply_logit_bias(
    logits: torch.Tensor,
    num_logit_bias: torch.Tensor,
    token_ids: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    num_reqs = logits.shape[0]
    max_num_logit_bias = token_ids.shape[-1]
    _logit_bias_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        num_logit_bias,
        token_ids,
        token_ids.stride(0),
        bias,
        bias.stride(0),
        BLOCK_SIZE=triton.next_power_of_2(max_num_logit_bias),
    )
