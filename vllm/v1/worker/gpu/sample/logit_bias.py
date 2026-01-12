# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _allowed_token_ids_kernel(
    logits_ptr,
    logits_stride,
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    n = tl.load(num_allowed_token_ids_ptr + req_idx)
    if n == 0:
        return

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < n
    allowed_tokens = tl.load(
        allowed_token_ids_ptr + req_idx * allowed_token_ids_stride + block, mask=mask
    )
    logits = tl.load(logits_ptr + req_idx * logits_stride + allowed_tokens, mask=mask)

    for i in range(0, vocab_size, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        tl.store(
            logits_ptr + req_idx * logits_stride + offset,
            float("-inf"),
            mask=offset < vocab_size,
        )

    tl.store(logits_ptr + req_idx * logits_stride + allowed_tokens, logits, mask=mask)


def apply_allowed_token_ids(
    logits: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
) -> None:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    assert allowed_token_ids.shape[-1] <= BLOCK_SIZE
    _allowed_token_ids_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        num_allowed_token_ids,
        allowed_token_ids,
        allowed_token_ids.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


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
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + req_idx * bias_stride + block, mask=mask)
    else:
        bias = float("-inf")

    logits = tl.load(logits_ptr + req_idx * logits_stride + token_ids, mask=mask)
    logits += bias
    tl.store(logits_ptr + req_idx * logits_stride + token_ids, logits, mask=mask)


def apply_logit_bias(
    logits: torch.Tensor,
    num_logit_bias: torch.Tensor,
    token_ids: torch.Tensor,
    bias: torch.Tensor | None,
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
        bias.stride(0) if bias is not None else 0,
        BLOCK_SIZE=triton.next_power_of_2(max_num_logit_bias),
    )


def apply_min_tokens(
    logits: torch.Tensor,
    pos: torch.Tensor,
    min_seq_len: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> None:
    num_stop_token_ids = num_stop_token_ids * (pos < min_seq_len)
    apply_logit_bias(
        logits,
        num_stop_token_ids,
        stop_token_ids,
        bias=None,
    )
