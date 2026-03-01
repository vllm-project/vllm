# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _min_p_kernel(
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    min_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
    if min_p == 0.0:
        return

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=float("-inf")
        )
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    threshold = max_val + tl.log(min_p)
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=float("-inf")
        )
        logits = tl.where(logits < threshold, float("-inf"), logits)
        tl.store(logits_ptr + req_idx * logits_stride + block, logits, mask=mask)


def apply_min_p(
    logits: torch.Tensor, idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    _min_p_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        idx_mapping,
        min_p,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
