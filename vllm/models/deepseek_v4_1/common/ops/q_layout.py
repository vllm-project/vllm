# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pad the local-head Q of the fused FlashMLA layout to the kernel's head count.

``wq_b`` is permuted so the Q GEMM writes each token as 32 chunks of
``[h_local, 16]``. The fused kernel wants 32 chunks of ``[padded_heads, 16]``
with zero padding heads (``"fused"``); the split-KV fallback wants the standard
``[padded_heads, 512]`` layout with GPT-J RoPE applied (``"standard_rope"``).
"""

from typing import Literal

import torch

from vllm.triton_utils import tl, triton

_HEAD_DIM = 512
_Q_CHUNK = 16
_NUM_CHUNKS = _HEAD_DIM // _Q_CHUNK
_ROPE_START_CHUNK = 448 // _Q_CHUNK


@triton.jit
def _q_layout_kernel(
    q_in,
    q_out,
    positions,
    cos_sin,
    COS_STRIDE: tl.constexpr,
    N_LOCAL: tl.constexpr,
    N_PADDED: tl.constexpr,
    ROPE_START_CHUNK: tl.constexpr,
    STANDARD_ROPE: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    offs = tl.arange(0, N_PADDED * 16)
    valid = offs < N_LOCAL * 16
    src = q_in + t * (N_LOCAL * 512) + c * (N_LOCAL * 16)
    x = tl.load(src + offs, mask=valid, other=0.0)
    if not STANDARD_ROPE:
        tl.store(q_out + t * (N_PADDED * 512) + c * (N_PADDED * 16) + offs, x)
    else:
        h = offs // 16
        j = offs % 16
        d = c * 16 + j
        if c >= ROPE_START_CHUNK:
            pos = tl.load(positions + t)
            pair = (d - ROPE_START_CHUNK * 16) // 2
            cos = tl.load(cos_sin + pos * COS_STRIDE + pair)
            sin = tl.load(cos_sin + pos * COS_STRIDE + 32 + pair)
            partner = tl.load(src + (offs ^ 1), mask=valid, other=0.0)
            xf = x.to(tl.float32)
            pf = partner.to(tl.float32)
            rotated = tl.where(j % 2 == 0, xf * cos - pf * sin, xf * cos + pf * sin)
            x = rotated.to(q_out.dtype.element_ty)
        tl.store(q_out + t * (N_PADDED * 512) + h * 512 + d, x)


def dsv41_q_layout(
    q: torch.Tensor,
    padded_heads: int,
    mode: Literal["fused", "standard_rope"],
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad fused-layout local Q to ``padded_heads`` (see module docstring).

    Args:
        q: ``[num_tokens, local_heads, 512]`` bf16 in the fused layout, each
            token contiguous.
        padded_heads: 64 or 128; must be >= ``local_heads``.
        mode: ``"fused"`` keeps the fused layout; ``"standard_rope"`` returns
            the standard layout with RoPE on the last 64 dims.
        positions: ``[num_tokens]`` int positions (``"standard_rope"`` only).
        cos_sin_cache: ``[max_pos, 64]`` fp32 (``"standard_rope"`` only).
    """
    num_tokens, local_heads, head_dim = q.shape
    assert head_dim == _HEAD_DIM and q.stride(2) == 1 and q.stride(1) == head_dim
    assert padded_heads >= local_heads and padded_heads % 16 == 0
    standard_rope = mode == "standard_rope"
    if not standard_rope and padded_heads == local_heads:
        return q
    out = torch.empty(
        (num_tokens, padded_heads, head_dim), dtype=q.dtype, device=q.device
    )
    cos_stride = 0
    if standard_rope:
        assert positions is not None and cos_sin_cache is not None
        assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.shape[1] == 64
        assert cos_sin_cache.stride(1) == 1
        cos_stride = cos_sin_cache.stride(0)
    else:
        positions, cos_sin_cache = out, out
    if num_tokens == 0:
        return out
    _q_layout_kernel[(num_tokens, _NUM_CHUNKS)](
        q,
        out,
        positions,
        cos_sin_cache,
        COS_STRIDE=cos_stride,
        N_LOCAL=local_heads,
        N_PADDED=padded_heads,
        ROPE_START_CHUNK=_ROPE_START_CHUNK,
        STANDARD_ROPE=standard_rope,
        num_warps=4,
    )
    return out
