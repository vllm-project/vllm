# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _p_less_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    order_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    # p-less sampling (https://arxiv.org/abs/2509.23234).
    #
    # Keep tokens whose probability is at least the order-k threshold
    #   T_k = (sum_v p(v) ** k) ** (1 / (k - 1)).
    # Working in logit space with m = max logit, S = sum exp(l - m) and
    # P_k = sum exp(k * (l - m)), the condition p(v) >= T_k reduces to
    #   l(v) >= m + (log(P_k) - log(S)) / (k - 1),
    # which avoids materializing the softmax and stays numerically stable.
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    order = tl.load(order_ptr + req_state_idx).to(tl.float32)
    if order == 0.0:
        # p-less disabled for this request.
        return

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    sum_exp = 0.0
    sum_exp_k = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        shifted = logits - max_val
        # Masked-out lanes were loaded as -inf, so their exp is 0 and does not
        # contribute to either sum.
        sum_exp += tl.sum(tl.exp(shifted))
        sum_exp_k += tl.sum(tl.exp(order * shifted))

    threshold = max_val + (tl.log(sum_exp_k) - tl.log(sum_exp)) / (order - 1.0)
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
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, order: torch.Tensor
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    _p_less_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        order,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
