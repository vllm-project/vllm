# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _p_less_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    p_less_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    p_less = tl.load(p_less_ptr + req_state_idx).to(tl.bool)
    if not p_less:
        return

    max_logit = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        max_logit = tl.max(tl.maximum(logits, max_logit))
    max_logit = max_logit.to(tl.float32)  # type: ignore

    sum_exps = 0.0
    sum_squared_exps = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        exps = tl.exp(logits - max_logit)
        sum_exps += tl.sum(exps)
        sum_squared_exps += tl.sum(exps**2.0)
    threshold = tl.log(sum_squared_exps) - tl.log(sum_exps) + max_logit

    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        logits = tl.where(logits < threshold, float("-inf"), logits)
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_p_less(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, p_less: torch.Tensor
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    _p_less_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        p_less,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
