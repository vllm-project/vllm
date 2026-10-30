# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for RoPE re-anchoring (vllm/v1/worker/reanchor_rotary.py).

Property: a cached key rotated at absolute position ``p`` and then re-rotated
by the constant ``R(-D)`` equals the key rotated at position ``p - D``. So after
dropping the query clock by ``D`` and re-rotating every live key by ``R(-D)``,
every relative attention score is preserved exactly. Pure torch / CPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.reanchor_rotary import (
    RotarySpec,
    apply_inverse_rotary,
    collect_rotary_specs,
    init_reanchor_rotary_specs,
    rerotate_cached_keys,
)


def _spec(rotary_dim: int, base: float, is_neox: bool, scale: int = 1):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    return RotarySpec(inv_freq.double(), rotary_dim, is_neox, scale)


def _forward_rotary(x: torch.Tensor, pos: int, spec: RotarySpec) -> torch.Tensor:
    """Rotate ``x`` by +pos (vLLM rotary convention)."""
    rotary_dim, half = spec.rotary_dim, spec.rotary_dim // 2
    ang = float(pos) * spec.inv_freq
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    rot, pas = x[..., :rotary_dim], x[..., rotary_dim:]
    if spec.is_neox:
        x1, x2 = rot[..., :half], rot[..., half:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        o1, o2 = x1 * cos - x2 * sin, x2 * cos + x1 * sin
        out = torch.stack([o1, o2], dim=-1).flatten(-2)
    return torch.cat([out, pas], dim=-1) if pas.numel() else out


# (head_size, rotary_dim, base, is_neox)
_STYLES = [
    pytest.param(128, 128, 1e6, True, id="voxtral-decoder-neox"),
    pytest.param(64, 64, 1e6, False, id="voxtral-encoder-gptj"),
    pytest.param(128, 128, 1e4, True, id="mistral-neox-1e4"),
    pytest.param(96, 32, 1e4, True, id="partial-rotary"),
]
_POS_SHIFTS = [
    pytest.param(5000, 4096, id="p5000-D4096"),
    pytest.param(9000, 7000, id="p9000-D7000"),
    pytest.param(120000, 100000, id="p120000-D100000"),
]


@pytest.mark.parametrize("head_size, rotary_dim, base, is_neox", _STYLES)
@pytest.mark.parametrize("p, D", _POS_SHIFTS)
def test_reanchor_identity(head_size, rotary_dim, base, is_neox, p, D):
    """R(-D) . R(p) . k == R(p-D) . k, pass-through dims untouched."""
    torch.manual_seed(0)
    spec = _spec(rotary_dim, base, is_neox)
    k = torch.randn(4, head_size, dtype=torch.float32)
    reanchored = apply_inverse_rotary(_forward_rotary(k, p, spec), D, spec)
    target = _forward_rotary(k, p - D, spec)
    assert (reanchored - target).abs().max().item() < 1e-3


@pytest.mark.parametrize("head_size, rotary_dim, base, is_neox", _STYLES)
@pytest.mark.parametrize("p, D", _POS_SHIFTS)
def test_reanchor_preserves_relative_score(head_size, rotary_dim, base, is_neox, p, D):
    torch.manual_seed(0)
    spec = _spec(rotary_dim, base, is_neox)
    k = torch.randn(4, head_size, dtype=torch.float32)
    q = torch.randn(4, head_size, dtype=torch.float32)
    i, j = p + 60, p  # j sits within the window below query position i
    s_orig = (_forward_rotary(q, i, spec) * _forward_rotary(k, j, spec)).sum(-1)
    kj_re = apply_inverse_rotary(_forward_rotary(k, j, spec), D, spec)
    s_re = (_forward_rotary(q, i - D, spec) * kj_re).sum(-1)
    assert (s_orig - s_re).abs().max().item() < 1e-3


def _layer(head_size: int, num_blocks: int = 6, dtype=torch.float32):
    # Standardized KV view: (blocks, heads, block_size, 2 * head_size).
    kv = torch.randn(num_blocks, 2, 4, 2 * head_size, dtype=dtype)
    return SimpleNamespace(head_size=head_size, kv_cache=kv)


def test_rerotate_cached_keys_uses_each_layer_spec():
    """Each layer moves by its own RoPE and position scale; values and blocks
    outside the request's row are untouched."""
    torch.manual_seed(0)
    ctx = {"dec": _layer(128), "enc": _layer(64)}
    specs = {"dec": _spec(128, 1e4, True), "enc": _spec(64, 1e6, False, scale=4)}
    groups = [
        SimpleNamespace(layer_names=["dec"]),
        SimpleNamespace(layer_names=["enc"]),
    ]
    before = {name: layer.kv_cache.clone() for name, layer in ctx.items()}
    rows = [torch.tensor([1, 3]), torch.tensor([2])]
    d = 2048

    rerotate_cached_keys(ctx, groups, rows, d, specs)

    for name, row in (("dec", rows[0]), ("enc", rows[1])):
        spec, head_size = specs[name], ctx[name].head_size
        kv, old = ctx[name].kv_cache, before[name]
        expected = apply_inverse_rotary(
            old[row, ..., :head_size], spec.position_scale * d, spec
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
        {"dec": layer},
        [group],
        [torch.tensor([], dtype=torch.long)],
        2048,
        {"dec": _spec(128, 1e6, True)},
    )
    assert torch.equal(layer.kv_cache, before)


def test_rerotate_cached_keys_rejects_quantized_kv():
    layer = _layer(128)
    layer.kv_cache = layer.kv_cache.to(torch.float8_e4m3fn)
    group = SimpleNamespace(layer_names=["dec"])
    with pytest.raises(AssertionError, match="non-quantized"):
        rerotate_cached_keys(
            {"dec": layer},
            [group],
            [torch.tensor([1])],
            2048,
            {"dec": _spec(128, 1e6, True)},
        )


class _Attention(nn.Module):
    """Stand-in for a vLLM attention layer: only ``layer_name`` matters."""

    def __init__(self, layer_name: str):
        super().__init__()
        self.layer_name = layer_name


class _SelfAttention(nn.Module):
    def __init__(self, prefix: str, rotary_emb: nn.Module, scale: int | None = None):
        super().__init__()
        self.rotary_emb = rotary_emb
        self.attn = _Attention(f"{prefix}.attn")
        if scale is not None:
            self.reanchor_position_scale = scale


def _model(*self_attns: _SelfAttention) -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList(self_attns)
    return model


_ROPES = [
    pytest.param({"rope_theta": 1e6}, id="default"),
    pytest.param(
        {
            "rope_type": "llama3",
            "rope_theta": 5e5,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 1024,
        },
        id="llama3",
    ),
    pytest.param(
        {
            "rope_type": "yarn",
            "rope_theta": 1e4,
            "factor": 4.0,
            "original_max_position_embeddings": 1024,
        },
        id="yarn",
    ),
]


@pytest.mark.parametrize("rope_parameters", _ROPES)
@pytest.mark.usefixtures("default_vllm_config")
def test_collect_rotary_specs_matches_forward_rope(rope_parameters):
    """The spec read from the model rotates exactly like the layer's own RoPE
    cache: same angle at every position, same rotary dim and style."""
    rope = get_rope(
        128,
        max_position=4096,
        is_neox_style=True,
        rope_parameters=rope_parameters,
        dtype=torch.float32,
    )
    model = _model(_SelfAttention("layers.0", rope))
    spec = collect_rotary_specs(model, ["layers.0.attn"])["layers.0.attn"]

    assert (spec.rotary_dim, spec.is_neox, spec.position_scale) == (128, True, 1)
    mscale = getattr(rope, "mscale", 1.0)
    half = spec.rotary_dim // 2
    # Same float32 frequencies as the forward pass: the cache's angle at p is
    # p * inv_freq rounded to float32, so compare in float32 too.
    for p in (1, 777, 3000):
        cos_cache = rope.cos_sin_cache[p, :half] / mscale
        cos_spec = torch.cos(p * spec.inv_freq.float())
        assert torch.allclose(cos_spec, cos_cache, atol=1e-5), (
            f"angle mismatch at position {p}"
        )


@pytest.mark.usefixtures("default_vllm_config")
def test_collect_rotary_specs_position_scale_and_sharing():
    rope = get_rope(64, max_position=4096, is_neox_style=False, dtype=torch.float32)
    model = _model(
        _SelfAttention("layers.0", rope),
        _SelfAttention("layers.1", rope),
        _SelfAttention("encoder.0", rope, scale=4),
    )
    specs = collect_rotary_specs(
        model, ["layers.0.attn", "layers.1.attn", "encoder.0.attn"]
    )
    assert specs["layers.0.attn"] is specs["layers.1.attn"]
    assert specs["encoder.0.attn"].position_scale == 4
    assert not specs["encoder.0.attn"].is_neox


@pytest.mark.usefixtures("default_vllm_config")
def test_collect_rotary_specs_rejects_unsupported_rope():
    rope = get_rope(
        128,
        max_position=4096,
        rope_parameters={"rope_type": "dynamic", "factor": 2.0},
        dtype=torch.float32,
    )
    model = _model(_SelfAttention("layers.0", rope))
    with pytest.raises(ValueError, match="does not support"):
        collect_rotary_specs(model, ["layers.0.attn"])


@pytest.mark.usefixtures("default_vllm_config")
def test_collect_rotary_specs_rejects_layer_without_rope():
    model = _model(_SelfAttention("layers.0", get_rope(128, max_position=4096)))
    with pytest.raises(ValueError, match="no rotary embedding"):
        collect_rotary_specs(model, ["layers.0.attn", "mamba.0"])


def _kv_cache_config(spec) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layers.0.attn"], spec)],
    )


@pytest.mark.usefixtures("default_vllm_config")
def test_init_reanchor_rotary_specs_requires_sliding_window():
    model = _model(_SelfAttention("layers.0", get_rope(128, max_position=4096)))
    common = dict(block_size=16, num_kv_heads=2, head_size=128, dtype=torch.float32)

    specs = init_reanchor_rotary_specs(
        model, _kv_cache_config(SlidingWindowSpec(sliding_window=64, **common))
    )
    assert set(specs) == {"layers.0.attn"}

    with pytest.raises(ValueError, match="sliding window"):
        init_reanchor_rotary_specs(model, _kv_cache_config(FullAttentionSpec(**common)))
