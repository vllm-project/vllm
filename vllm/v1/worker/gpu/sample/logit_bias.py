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
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    n = tl.load(num_logit_bias_ptr + req_idx)
    if n == 0:
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + req_idx * logits_stride + block, mask=mask)

    # NOTE(woosuk): We assume n is small. If n is large, this loop will be slow.
    for i in range(n):
        token_id = tl.load(token_ids_ptr + req_idx * token_ids_stride + i)
        bias = tl.load(bias_ptr + req_idx * bias_stride + i)
        logits += bias * (block == token_id)
    tl.store(logits_ptr + req_idx * logits_stride + block, logits, mask=mask)


def apply_logit_bias(
    logits: torch.Tensor,
    num_logit_bias: torch.Tensor,
    token_ids: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _logit_bias_kernel[(num_reqs, num_blocks)](
        logits,
        logits.stride(0),
        num_logit_bias,
        token_ids,
        token_ids.stride(0),
        bias,
        bias.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
