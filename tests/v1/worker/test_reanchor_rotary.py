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

from types import SimpleNamespace

import pytest
import torch

# Import the PRODUCTION op so the test guards the real function, not a copy.
from vllm.v1.worker.reanchor_rotary import (
    ROPE_THETA,
    apply_inverse_rotary,
    audio_position_factor,
    rerotate_cached_keys,
)


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


def _layer(head_size: int, num_blocks: int = 6, dtype=torch.float32):
    # Standardized KV view: (blocks, heads, block_size, 2 * head_size).
    kv = torch.randn(num_blocks, 2, 4, 2 * head_size, dtype=dtype)
    return SimpleNamespace(head_size=head_size, kv_cache=kv)


def test_rerotate_cached_keys_touches_only_row_keys():
    """Only the keys of the listed blocks move: decoder by d, encoder by
    audio_factor * d. Values and other blocks are untouched."""
    torch.manual_seed(0)
    decoder, encoder = _layer(128), _layer(64)
    ctx = {"dec": decoder, "enc": encoder}
    groups = [
        SimpleNamespace(layer_names=["dec"]),
        SimpleNamespace(layer_names=["enc"]),
    ]
    before = {name: layer.kv_cache.clone() for name, layer in ctx.items()}
    rows = [torch.tensor([1, 3]), torch.tensor([2])]
    d, audio_factor = 2048, 4

    rerotate_cached_keys(ctx, groups, rows, d, audio_factor)

    for name, row, head_size, shift, is_neox in (
        ("dec", rows[0], 128, d, True),
        ("enc", rows[1], 64, audio_factor * d, False),
    ):
        kv, old = ctx[name].kv_cache, before[name]
        expected = apply_inverse_rotary(
            old[row, ..., :head_size], shift, head_size, ROPE_THETA, is_neox
        )
        assert torch.allclose(kv[row, ..., :head_size], expected, atol=1e-5)
        assert torch.equal(kv[..., head_size:], old[..., head_size:])
        others = [b for b in range(kv.shape[0]) if b not in row.tolist()]
        assert torch.equal(kv[others], old[others])


def test_rerotate_cached_keys_skips_empty_rows():
    layer = _layer(128)
    before = layer.kv_cache.clone()
    group = SimpleNamespace(layer_names=["dec"])
    rerotate_cached_keys(
        {"dec": layer}, [group], [torch.tensor([], dtype=torch.long)], 2048, 4
    )
    assert torch.equal(layer.kv_cache, before)


def test_rerotate_cached_keys_rejects_quantized_kv():
    layer = _layer(128)
    layer.kv_cache = layer.kv_cache.to(torch.float8_e4m3fn)
    group = SimpleNamespace(layer_names=["dec"])
    with pytest.raises(AssertionError, match="non-quantized"):
        rerotate_cached_keys({"dec": layer}, [group], [torch.tensor([1])], 2048, 4)


def test_audio_position_factor():
    def config(**audio):
        return SimpleNamespace(audio_config=SimpleNamespace(**audio))

    assert audio_position_factor(config(block_pool_size=4)) == 4
    assert audio_position_factor(config(downsample_factor=4)) == 4
    assert audio_position_factor(SimpleNamespace()) == 1
