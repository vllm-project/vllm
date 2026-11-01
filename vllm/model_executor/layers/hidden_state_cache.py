# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Masked cache writes with stable buffers for CUDA graph replay."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _store_window(
    H,
    C,
    S,
    R,
    H0: tl.constexpr,
    H1: tl.constexpr,
    H2: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    token = tl.program_id(0)
    part = tl.program_id(1)
    layer = part // tl.cdiv(WIDTH, TILE)
    columns = part % tl.cdiv(WIDTH, TILE) * TILE + tl.arange(0, TILE)
    slot = tl.load(S + token).to(tl.int64)
    block = tl.maximum(slot, 0) // BLOCK
    offset = tl.maximum(slot, 0) % BLOCK
    lo = tl.load(R + block * 2)
    hi = tl.load(R + block * 2 + 1)
    valid = (slot >= 0) & (offset >= lo) & (offset < hi) & (columns < WIDTH)
    value = tl.load(
        H + token.to(tl.int64) * H0 + layer * H1 + columns * H2, valid, other=0
    )
    tl.store(C + block * C0 + offset * C1 + layer * C2 + columns * C3, value, valid)


def store_hidden_state_window(
    hidden: torch.Tensor, cache: torch.Tensor, slots: torch.Tensor, ranges: torch.Tensor
) -> None:
    if hidden.shape[0] == 0:
        return
    width = hidden.shape[2]
    _store_window[(hidden.shape[0], hidden.shape[1] * triton.cdiv(width, 512))](
        hidden,
        cache,
        slots,
        ranges,
        *hidden.stride(),
        *cache.stride(),
        hidden.shape[1],
        width,
        cache.shape[1],
        512,
    )
