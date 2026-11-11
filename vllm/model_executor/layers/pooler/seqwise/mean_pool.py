# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _segmented_mean_kernel(
    x,
    starts,
    ends,
    out,
    stride_t,
    stride_h,
    stride_start,
    stride_end,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    seq = tl.program_id(0)
    start = tl.load(starts + seq * stride_start).to(tl.int64)
    length = tl.load(ends + seq * stride_end).to(tl.int64) + 1 - start
    rows = tl.arange(0, BLOCK_T)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.full((BLOCK_T, BLOCK_H), 0, tl.float32)
    for offset in range(0, length, BLOCK_T):
        tokens = offset + rows
        values = tl.load(
            x + (start + tokens[:, None]) * stride_t + cols[None, :] * stride_h,
            mask=(tokens[:, None] < length) & (cols[None, :] < H),
            other=0,
        ).to(tl.float32)
        acc += values
    mean = tl.sum(acc, axis=0) / length.to(tl.float32)
    tl.store(out + seq * H + cols, mean, mask=cols < H)


def _segmented_mean(
    hidden_states: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """Average contiguous token ranges with inclusive ends into FP32 rows."""
    num_seqs = starts.numel()
    hidden_size = hidden_states.shape[1]
    out = torch.empty(
        (num_seqs, hidden_size), device=hidden_states.device, dtype=torch.float32
    )
    if num_seqs and hidden_size:
        _segmented_mean_kernel[(num_seqs, triton.cdiv(hidden_size, 128))](
            hidden_states,
            starts,
            ends,
            out,
            hidden_states.stride(0),
            hidden_states.stride(1),
            starts.stride(0),
            ends.stride(0),
            H=hidden_size,
            BLOCK_T=32,
            BLOCK_H=128,
            num_warps=4,
        )
    return out


def _segmented_mean_fake(
    hidden_states: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    return hidden_states.new_empty(
        (starts.numel(), hidden_states.shape[1]), dtype=torch.float32
    )


direct_register_custom_op(
    op_name="segmented_mean",
    op_func=_segmented_mean,
    mutates_args=[],
    fake_impl=_segmented_mean_fake,
)


def segmented_mean(
    hidden_states: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    return torch.ops.vllm.segmented_mean(hidden_states, starts, ends)
