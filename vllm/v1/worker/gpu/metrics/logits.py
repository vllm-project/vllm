# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch._inductor.runtime.triton_helpers import libdevice

from vllm.triton_utils import tl, triton


@triton.jit
def _num_nans_kernel(
    logits_ptr,
    logits_stride,
    num_nans_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_nans = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=0
        )
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int1)
        num_nans += tl.sum(is_nan).to(tl.int32)
    tl.store(num_nans_ptr + req_idx, num_nans)


def get_num_nans(logits: torch.Tensor) -> torch.Tensor:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _num_nans_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        num_nans,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return num_nans


def aggregate_num_nans_per_request(
    num_nans: torch.Tensor,
    cumulative_row_ends: torch.Tensor,
) -> torch.Tensor:
    """Aggregate per-logit-row counts for speculative requests on device."""
    prefix_sum = torch.empty(
        num_nans.shape[0] + 1, dtype=num_nans.dtype, device=num_nans.device
    )
    prefix_sum[0] = 0
    torch.cumsum(num_nans, dim=0, out=prefix_sum[1:])

    row_starts = torch.empty_like(cumulative_row_ends)
    row_starts[0] = 0
    row_starts[1:] = cumulative_row_ends[:-1]
    return prefix_sum[cumulative_row_ends] - prefix_sum[row_starts]
