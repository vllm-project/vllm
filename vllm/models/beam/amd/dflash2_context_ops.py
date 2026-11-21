# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["ROWS"])
def _transform_context_kv_kernel(
    KV,
    WEIGHT,
    POSITIONS,
    ROPE,
    KEY_OUT,
    VALUE_OUT,
    ROWS,
    LAYERS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    VALUE_SCALE: tl.constexpr,
    NEOX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    layer_head = tl.program_id(1)
    layer = layer_head // HEADS
    head = layer_head % HEADS
    col = tl.arange(0, BLOCK)
    mask = (row[:, None] < ROWS) & (col[None, :] < DIM)
    kv_offsets = (
        row[:, None] * (LAYERS * 2 * HEADS * DIM)
        + (layer * 2 * HEADS + head) * DIM
        + col[None, :]
    )
    key = tl.load(KV + kv_offsets, mask, 0).to(tl.float32)
    if BLOCK == 128:
        squares = (key * key).reshape((4, 4, 32))
        total = tl.sum(tl.sum(squares, 2), 1)
    else:
        total = tl.sum(key * key, 1)
    weight = tl.load(WEIGHT + layer * DIM + col, col < DIM, 0).to(tl.float32)
    key = key * tl.rsqrt(total / DIM + EPS)[:, None] * weight[None, :]
    # Beam rounds K normalization before RoPE.
    key = key.to(KV.dtype.element_ty).to(tl.float32)
    half: tl.constexpr = DIM // 2
    if NEOX:
        partner = (col + half) % DIM
        second = col >= half
        rope_col = col % half
    else:
        partner = col ^ 1
        second = col % 2 != 0
        rope_col = col // 2
    other = tl.gather(key, tl.broadcast_to(partner[None, :], (4, BLOCK)), 1)
    position = tl.load(POSITIONS + row, row < ROWS, 0).to(tl.int64)
    cosine = tl.load(ROPE + position[:, None] * DIM + rope_col[None, :], mask, 0)
    sine = tl.load(ROPE + position[:, None] * DIM + half + rope_col[None, :], mask, 0)
    key = tl.where(
        second[None, :],
        tl.fma(other, sine, key * cosine),
        tl.fma(key, cosine, -other * sine),
    )
    out_offsets = (
        (layer * ROWS + row[:, None]) * (HEADS * DIM) + head * DIM + col[None, :]
    )
    value = tl.load(KV + kv_offsets + HEADS * DIM, mask, 0).to(tl.float32)
    tl.store(KEY_OUT + out_offsets, key, mask)
    tl.store(VALUE_OUT + out_offsets, value * VALUE_SCALE, mask)


def transform_context_kv(
    kv: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    rope: torch.Tensor,
    eps: float,
    value_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Beam K normalization/RoPE and V scaling without touching caches."""
    rows, layers, _, heads, dim = kv.shape
    keys = torch.empty((layers, rows, heads, dim), dtype=kv.dtype, device=kv.device)
    values = torch.empty_like(keys)
    if rows:
        _transform_context_kv_kernel[(triton.cdiv(rows, 4), layers * heads)](
            kv,
            weight,
            positions,
            rope,
            keys,
            values,
            rows,
            layers,
            heads,
            dim,
            eps,
            value_scale,
            is_neox,
            triton.next_power_of_2(dim),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return keys, values
