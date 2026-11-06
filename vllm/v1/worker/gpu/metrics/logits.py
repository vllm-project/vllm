# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch._inductor.runtime.triton_helpers import libdevice

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_logits"])
def _num_nans_kernel(
    logits_ptr,
    logits_stride,
    cu_num_logits_ptr,
    num_nans_ptr,
    num_logits,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (request, vocab chunk); the chunks of a request
    # accumulate into its slot with atomics, so num_nans must start zeroed.
    req_idx = tl.program_id(0)
    block = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    row_start = tl.load(cu_num_logits_ptr + req_idx)
    row_end = tl.minimum(tl.load(cu_num_logits_ptr + req_idx + 1), num_logits)
    num_nans = tl.zeros((), dtype=tl.int32)
    for row in range(row_start, row_end):
        logits = tl.load(
            logits_ptr + row.to(tl.int64) * logits_stride + block,
            mask=mask,
            other=0,
        )
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int1)
        num_nans += tl.sum(is_nan).to(tl.int32)
    tl.atomic_add(num_nans_ptr + req_idx, num_nans)


def get_num_nans(logits: torch.Tensor, cu_num_logits: torch.Tensor) -> torch.Tensor:
    """Count NaNs per request over its logits rows.

    Args:
        logits: [num_logits, vocab_size]. Request j owns rows
            ``cu_num_logits[j]:cu_num_logits[j + 1]``.
        cu_num_logits: [num_reqs + 1] row offsets on the same device.

    Returns:
        [num_reqs] int32 NaN counts.

    """
    num_reqs = cu_num_logits.shape[0] - 1
    vocab_size = logits.shape[1]
    BLOCK_SIZE = 2048
    num_nans = torch.zeros(num_reqs, dtype=torch.int32, device=logits.device)
    _num_nans_kernel[(num_reqs, triton.cdiv(vocab_size, BLOCK_SIZE))](
        logits,
        logits.stride(0),
        cu_num_logits,
        num_nans,
        logits.shape[0],
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return num_nans
