# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode reduce epilogue must reproduce the standalone inverse RoPE.

Guards the lane arithmetic of the fused epilogue in
``_sparse_attn_decode_reduce_kernel``: it rotates a whole [H, nope + rope] row
with one expression by feeding cos=1/sin=0 to the NoPE lanes, which only works
if the pair-split boundary lands exactly on ``nope_head_dim // 2``.
"""

import pytest
import torch

NOPE = 448
ROPE = 64
COMB = NOPE + ROPE


def _reference_inverse_rope(o, positions, cos_sin_cache):
    """What ``_inverse_rope_gptj_kernel`` computes, lane by lane."""
    out = o.clone()
    cos = cos_sin_cache[positions, : ROPE // 2]
    sin = cos_sin_cache[positions, ROPE // 2 :]
    a = o[..., NOPE::2]
    b = o[..., NOPE + 1 :: 2]
    out[..., NOPE::2] = a * cos[:, None, :] + b * sin[:, None, :]
    out[..., NOPE + 1 :: 2] = b * cos[:, None, :] - a * sin[:, None, :]
    return out


def _fused_epilogue(o, positions, cos_sin_cache):
    """The tile arithmetic of the fused epilogue, in torch."""
    pair_idx = torch.arange(COMB // 2) - NOPE // 2
    is_rope = pair_idx >= 0
    k = torch.where(is_rope, pair_idx, torch.zeros_like(pair_idx))
    row = cos_sin_cache[positions]
    cos = torch.where(is_rope, row[:, k], torch.ones(1))
    sin = torch.where(is_rope, row[:, ROPE // 2 + k], torch.zeros(1))
    even, odd = o.reshape(*o.shape[:-1], COMB // 2, 2).unbind(-1)
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return torch.stack(
        (even * cos + odd * sin, odd * cos - even * sin), dim=-1
    ).reshape(o.shape)


@pytest.mark.parametrize("num_heads", [1, 8])
def test_fused_epilogue_matches_standalone_inverse_rope(num_heads):
    torch.manual_seed(0)
    num_tokens, max_pos = 7, 64
    o = torch.randn(num_tokens, num_heads, COMB, dtype=torch.float32)
    positions = torch.randint(0, max_pos, (num_tokens,))
    cos_sin_cache = torch.randn(max_pos, ROPE, dtype=torch.float32)

    fused = _fused_epilogue(o, positions, cos_sin_cache)
    torch.testing.assert_close(
        fused, _reference_inverse_rope(o, positions, cos_sin_cache)
    )
    # NoPE lanes must come through bit-exact, not merely close.
    assert torch.equal(fused[..., :NOPE], o[..., :NOPE])
