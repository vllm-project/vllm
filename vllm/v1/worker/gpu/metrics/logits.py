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


@triton.jit
def _aggregate_num_nans_per_request_kernel(
    num_nans_ptr,
    num_nans_stride,
    cumulative_row_ends_ptr,
    cumulative_row_ends_stride,
    result_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum one request's rows from the previous end to its own end.

    Counts [1, 2, 3, 4, 5] with ends [2, 3, 5] yield [3, 3, 9].
    Counts [1, 2, 3] with ends [2, 2, 3] yield [3, 0, 3].
    """
    req_idx = tl.program_id(0)
    start = tl.load(
        cumulative_row_ends_ptr + (req_idx - 1) * cumulative_row_ends_stride,
        mask=req_idx > 0,
        other=0,
    )
    end = tl.load(cumulative_row_ends_ptr + req_idx * cumulative_row_ends_stride)
    total = 0
    for offset in range(start, end, BLOCK_SIZE):
        rows = offset + tl.arange(0, BLOCK_SIZE)
        counts = tl.load(
            num_nans_ptr + rows * num_nans_stride, mask=rows < end, other=0
        )
        total += tl.sum(counts)
    tl.store(result_ptr + req_idx, total)


def aggregate_num_nans_per_request(
    num_nans: torch.Tensor,
    cumulative_row_ends: torch.Tensor,
) -> torch.Tensor:
    """Aggregate per-logit-row counts for speculative requests on device."""
    result = torch.empty(
        cumulative_row_ends.numel(), dtype=num_nans.dtype, device=num_nans.device
    )
    if result.numel() == 0:
        return result
    _aggregate_num_nans_per_request_kernel[(cumulative_row_ends.numel(),)](
        num_nans,
        num_nans.stride(0),
        cumulative_row_ends,
        cumulative_row_ends.stride(0),
        result,
        BLOCK_SIZE=32,
        num_warps=1,
    )
    return result
