# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage BF16 tokens and routing into MegaMoE's symmetric buffer."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_bf16_kernel(
    hidden,
    weights,
    ids,
    x_out,
    ids_out,
    weights_out,
    padding,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    stride_i: tl.constexpr,
    stride_p: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    x = tl.load(hidden + token * stride_h + cols, mask=cols < H, other=0)
    tl.store(x_out + token * H + cols, x, mask=cols < H)
    routes = tl.arange(0, BLOCK_K)
    idx = tl.load(ids + token * stride_i + routes, mask=routes < K, other=-1)
    weight = tl.load(weights + token * stride_w + routes, mask=routes < K, other=0)
    if padding is not None:
        padded = tl.load(padding + token * stride_p)
        idx = tl.where(padded, -1, idx)
        weight = tl.where(padded, 0.0, weight)
    tl.store(ids_out + token * K + routes, idx.to(tl.int64), mask=routes < K)
    tl.store(weights_out + token * K + routes, weight, mask=routes < K)


def prepare_megamoe_bf16_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_out: torch.Tensor,
    ids_out: torch.Tensor,
    weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> None:
    tokens, hidden = hidden_states.shape
    assert hidden_states.dtype == x_out.dtype == torch.bfloat16
    assert hidden_states.stride(1) == topk_ids.stride(1) == topk_weights.stride(1) == 1
    assert (
        x_out.is_contiguous()
        and ids_out.is_contiguous()
        and weights_out.is_contiguous()
    )
    assert x_out.shape == hidden_states.shape
    assert topk_ids.shape == topk_weights.shape == ids_out.shape == weights_out.shape
    assert topk_ids.shape[0] == tokens
    if tokens == 0:
        return
    _prepare_bf16_kernel[(tokens,)](
        hidden_states,
        topk_weights,
        topk_ids,
        x_out,
        ids_out,
        weights_out,
        is_padding,
        hidden_states.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        is_padding.stride(0) if is_padding is not None else 0,
        hidden,
        topk_ids.shape[1],
        triton.next_power_of_2(hidden),
        triton.next_power_of_2(topk_ids.shape[1]),
    )
