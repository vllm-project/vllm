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


@triton.jit
def _num_nans_spec_decode_kernel(
    logits_ptr,
    logits_stride,
    cu_num_logits_ptr,
    num_nans_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_logits = end_idx - start_idx

    num_nans = 0
    for row_offset in range(num_logits):
        row_idx = start_idx + row_offset
        for i in range(0, vocab_size, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < vocab_size
            logits = tl.load(
                logits_ptr + row_idx * logits_stride + block,
                mask=mask,
                other=0,
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


def get_num_nans_spec_decode(
    logits: torch.Tensor,
    cu_num_logits: torch.Tensor,
) -> torch.Tensor:
    num_reqs = cu_num_logits.shape[0] - 1
    vocab_size = logits.shape[1]
    BLOCK_SIZE = 8192
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _num_nans_spec_decode_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        cu_num_logits,
        num_nans,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return num_nans
