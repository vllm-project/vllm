# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collected-in-CI unit test for the RoPE re-anchoring core math.

The manual harness benchmarks/voxtral_realtime/test_reanchor_math.py is a
``__main__`` script that pytest never collects. This mirrors it as a real
pytest case so the production re-anchor op stays guarded on every run. Pure
torch / CPU: no GPU, no model load.

Property: a POST-rotation cached key rotated at absolute position ``p`` and
then re-rotated by the constant ``R(-D)`` equals the key rotated at position
``p - D``. So after dropping the query clock by ``D`` and re-rotating every live
key by ``R(-D)``, every in-window relative attention score ``R(i-j)`` is
preserved exactly. Checked for both styles used by Voxtral realtime: the NeoX
decoder (rotary_dim=128) and the GPT-J audio encoder (rotary_dim=64), base=1e6.
"""

import pytest
import torch

# Import the PRODUCTION op so the test guards the real function, not a copy.
from vllm.v1.worker.reanchor_rotary import apply_inverse_rotary


def _inv_freq(rotary_dim: int, base: float) -> torch.Tensor:
    return 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )


def _forward_rotary(
    x: torch.Tensor, pos: int, rotary_dim: int, base: float, is_neox: bool
) -> torch.Tensor:
    """Rotate ``x`` by +pos (vLLM rotary convention)."""
    half = rotary_dim // 2
    ang = float(pos) * _inv_freq(rotary_dim, base)
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    rot, pas = x[..., :rotary_dim], x[..., rotary_dim:]
    if is_neox:
        x1, x2 = rot[..., :half], rot[..., half:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        o1, o2 = x1 * cos - x2 * sin, x2 * cos + x1 * sin
        out = torch.stack([o1, o2], dim=-1).flatten(-2)
    return torch.cat([out, pas], dim=-1) if pas.numel() else out


# (rotary_dim, base, is_neox) for the two Voxtral-realtime rotary styles.
_STYLES = [
    pytest.param(128, 1e6, True, id="decoder-neox-128"),
    pytest.param(64, 1e6, False, id="encoder-gptj-64"),
]
# (absolute position p, re-anchor distance D), incl. a near-max_model_len case.
_POS_SHIFTS = [
    pytest.param(5000, 4096, id="p5000-D4096"),
    pytest.param(9000, 7000, id="p9000-D7000"),
    pytest.param(3000, 2048, id="p3000-D2048"),
    pytest.param(120000, 100000, id="p120000-D100000"),
]


@pytest.mark.parametrize("rotary_dim, base, is_neox", _STYLES)
@pytest.mark.parametrize("p, D", _POS_SHIFTS)
def test_reanchor_identity(rotary_dim, base, is_neox, p, D):
    """R(-D) . R(p) . k == R(p-D) . k for the production op."""
    torch.manual_seed(0)
    k = torch.randn(4, rotary_dim, dtype=torch.float32)
    cached = _forward_rotary(k, p, rotary_dim, base, is_neox)  # key in KV cache
    reanchored = apply_inverse_rotary(cached, D, rotary_dim, base, is_neox)
    target = _forward_rotary(k, p - D, rotary_dim, base, is_neox)
    assert (reanchored - target).abs().max().item() < 1e-3


@pytest.mark.parametrize("rotary_dim, base, is_neox", _STYLES)
@pytest.mark.parametrize("p, D", _POS_SHIFTS)
def test_reanchor_preserves_relative_score(rotary_dim, base, is_neox, p, D):
    """In-window q@k score is unchanged after dropping the clock by D and
    re-rotating the key by R(-D)."""
    torch.manual_seed(0)
    k = torch.randn(4, rotary_dim, dtype=torch.float32)
    q = torch.randn(4, rotary_dim, dtype=torch.float32)
    i, j = p + 60, p  # j sits within the window below query position i
    s_orig = (
        _forward_rotary(q, i, rotary_dim, base, is_neox)
        * _forward_rotary(k, j, rotary_dim, base, is_neox)
    ).sum(-1)
    kj_re = apply_inverse_rotary(
        _forward_rotary(k, j, rotary_dim, base, is_neox),
        D,
        rotary_dim,
        base,
        is_neox,
    )
    s_re = (_forward_rotary(q, i - D, rotary_dim, base, is_neox) * kj_re).sum(-1)
    assert (s_orig - s_re).abs().max().item() < 1e-3
