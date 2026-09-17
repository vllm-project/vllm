# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decode reduce epilogue must reproduce the standalone inverse RoPE.

The epilogue in ``_sparse_attn_decode_reduce_kernel`` rotates a whole
[H, nope + rope] row with one expression by feeding cos=1/sin=0 to the NoPE
lanes, which only holds if the pair split lands on ``nope_head_dim // 2``.
"""

import torch

NOPE = 448
ROPE = 64
COMB = NOPE + ROPE


def _standalone(o, positions, cos_sin_cache):
    """What ``_inverse_rope_gptj_kernel`` computes, lane by lane."""
    out = o.clone()
    cos = cos_sin_cache[positions, : ROPE // 2][:, None, :]
    sin = cos_sin_cache[positions, ROPE // 2 :][:, None, :]
    a = o[..., NOPE::2]
    b = o[..., NOPE + 1 :: 2]
    out[..., NOPE::2] = a * cos + b * sin
    out[..., NOPE + 1 :: 2] = b * cos - a * sin
    return out


def _epilogue(o, positions, cos_sin_cache):
    """The tile arithmetic of the fused epilogue, in torch."""
    pair_idx = torch.arange(COMB // 2) - NOPE // 2
    is_rope = pair_idx >= 0
    k = torch.where(is_rope, pair_idx, torch.zeros_like(pair_idx))
    row = cos_sin_cache[positions]
    cos = torch.where(is_rope, row[:, k], torch.ones(1))[:, None, :]
    sin = torch.where(is_rope, row[:, ROPE // 2 + k], torch.zeros(1))[:, None, :]
    even, odd = o.reshape(*o.shape[:-1], COMB // 2, 2).unbind(-1)
    return torch.stack(
        (even * cos + odd * sin, odd * cos - even * sin), dim=-1
    ).reshape(o.shape)


def test_fused_epilogue_matches_standalone_inverse_rope():
    torch.manual_seed(0)
    num_tokens, num_heads, max_pos = 7, 8, 64
    o = torch.randn(num_tokens, num_heads, COMB, dtype=torch.float32)
    positions = torch.randint(0, max_pos, (num_tokens,))
    cos_sin_cache = torch.randn(max_pos, ROPE, dtype=torch.float32)

    fused = _epilogue(o, positions, cos_sin_cache)
    torch.testing.assert_close(fused, _standalone(o, positions, cos_sin_cache))
    # NoPE lanes must come through bit-exact, not merely close.
    assert torch.equal(fused[..., :NOPE], o[..., :NOPE])
