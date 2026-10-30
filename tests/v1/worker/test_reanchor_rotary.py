# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm/v1/worker/reanchor_rotary.py.

A key rotated at position ``p`` and then by ``R(-d)`` equals the key rotated at
``p - d``. Query and keys shift together, so attention scores do not change.
"""

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVQuantMode,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.reanchor_rotary import (
    RotarySpec,
    apply_inverse_rotary,
    collect_rotary_specs,
    init_reanchor_rotary_specs,
    rerotate_cached_keys,
)

pytestmark = pytest.mark.cpu_test


def _spec(rotary_dim: int, base: float, is_neox: bool, scale: int = 1):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    return RotarySpec(inv_freq.double(), rotary_dim, is_neox, scale)


def _forward_rotary(x: torch.Tensor, pos: int, spec: RotarySpec) -> torch.Tensor:
    """Rotate ``x`` by +pos, as the forward pass does."""
    angle = pos * spec.inv_freq
    rotated = ApplyRotaryEmb.forward_static(
        x[..., : spec.rotary_dim], angle.cos(), angle.sin(), spec.is_neox
    )
    return torch.cat((rotated, x[..., spec.rotary_dim :]), dim=-1)


# (head_size, rotary_dim, base, is_neox)
_STYLES = [
    pytest.param(128, 128, 1e6, True, id="voxtral-decoder-neox"),
    pytest.param(64, 64, 1e6, False, id="voxtral-encoder-gptj"),
    pytest.param(128, 128, 1e4, True, id="mistral-neox-1e4"),
    pytest.param(96, 32, 1e4, True, id="partial-rotary"),
]
_POS_SHIFTS = [
    pytest.param(5000, 4096, id="p5000-d4096"),
    pytest.param(9000, 7000, id="p9000-d7000"),
    pytest.param(120000, 100000, id="p120000-d100000"),
]


@pytest.mark.parametrize("head_size, rotary_dim, base, is_neox", _STYLES)
@pytest.mark.parametrize("p, d", _POS_SHIFTS)
def test_reanchor_identity(head_size, rotary_dim, base, is_neox, p, d):
    """R(-d) . R(p) . k == R(p-d) . k, pass-through dims untouched."""
    torch.manual_seed(0)
    spec = _spec(rotary_dim, base, is_neox)
    k = torch.randn(4, head_size, dtype=torch.float32)
    reanchored = apply_inverse_rotary(_forward_rotary(k, p, spec), d, spec)
    target = _forward_rotary(k, p - d, spec)
    assert (reanchored - target).abs().max().item() < 1e-3


def _attention(prefix: str, head_size: int = 128, **kwargs) -> Attention:
    return Attention(
        num_heads=2, head_size=head_size, scale=1.0, prefix=prefix, **kwargs
    )


def _bind_kv_cache(attn: Attention, num_blocks: int = 8) -> None:
    # Layer view: (blocks, heads, block_size, 2 * head_size).
    attn.bind_kv_cache(torch.randn(num_blocks, 2, 4, 2 * attn.head_size))


def test_rerotate_cached_keys_uses_each_layer_spec(default_vllm_config):
    """Each layer moves by its own RoPE and position scale. Values and the blocks
    outside the request's row are untouched."""
    torch.manual_seed(0)
    dec, enc = _attention("dec"), _attention("enc", head_size=64)
    _bind_kv_cache(dec)
    _bind_kv_cache(enc)
    layers = {"dec": dec, "enc": enc}
    specs = {"dec": _spec(128, 1e4, True), "enc": _spec(64, 1e6, False, scale=4)}
    window = SlidingWindowSpec(
        sliding_window=64,
        block_size=4,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.float32,
    )
    groups = [KVCacheGroupSpec(["dec"], window), KVCacheGroupSpec(["enc"], window)]
    before = {name: layer.kv_cache.clone() for name, layer in layers.items()}
    rows = [torch.tensor([1, 3]), torch.tensor([2])]
    d = 2048

    rerotate_cached_keys(
        default_vllm_config.compilation_config.static_forward_context,
        groups,
        rows,
        d,
        specs,
    )

    for name, row in (("dec", rows[0]), ("enc", rows[1])):
        spec, head_size = specs[name], layers[name].head_size
        kv, old = layers[name].kv_cache, before[name]
        expected = apply_inverse_rotary(
            old[row, ..., :head_size], spec.position_scale * d, spec
        )
        assert torch.allclose(kv[row, ..., :head_size], expected, atol=1e-5)
        assert torch.equal(kv[..., head_size:], old[..., head_size:])
        others = [b for b in range(kv.shape[0]) if b not in row.tolist()]
        assert torch.equal(kv[others], old[others])


class _SelfAttention(nn.Module):
    """A vLLM attention layer with its RoPE on the parent module."""

    def __init__(self, prefix: str, rotary_emb: nn.Module, scale: int | None = None):
        super().__init__()
        self.rotary_emb = rotary_emb
        self.attn = _attention(f"{prefix}.attn", head_size=rotary_emb.head_size)
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
    """The spec's frequencies give the same angles as the layer's RoPE cache."""
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
    # The cache's angle at p is p * inv_freq rounded to float32: compare in float32.
    for p in (1, 777, 3000):
        cos_cache = rope.cos_sin_cache[p, :half] / mscale
        cos_spec = torch.cos(p * spec.inv_freq.float())
        assert torch.allclose(cos_spec, cos_cache, atol=1e-5), (
            f"angle mismatch at position {p}"
        )


@pytest.mark.usefixtures("default_vllm_config")
def test_collect_rotary_specs_position_scale():
    rope = get_rope(64, max_position=4096, is_neox_style=False, dtype=torch.float32)
    model = _model(
        _SelfAttention("layers.0", rope),
        _SelfAttention("encoder.0", rope, scale=4),
    )
    specs = collect_rotary_specs(model, ["layers.0.attn", "encoder.0.attn"])
    assert specs["layers.0.attn"].position_scale == 1
    assert specs["encoder.0.attn"].position_scale == 4


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
    with pytest.raises(ValueError, match="no rotary_emb"):
        collect_rotary_specs(model, ["layers.0.attn", "mamba.0"])


@pytest.mark.usefixtures("default_vllm_config")
def test_collect_rotary_specs_rejects_rope_on_part_of_the_key():
    self_attn = nn.Module()
    self_attn.rotary_emb = get_rope(64, max_position=4096)
    self_attn.attn = _attention("layers.0.attn", head_size=128)
    with pytest.raises(ValueError, match="part of its keys"):
        collect_rotary_specs(self_attn, ["layers.0.attn"])


_SPEC_ARGS = dict(block_size=4, num_kv_heads=2, head_size=128, dtype=torch.float32)


def _kv_cache_config(spec, layer_names=("layers.0.attn",)) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(layer_names), spec)],
    )


def test_init_reanchor_rotary_specs_returns_each_layer(default_vllm_config):
    model = _model(_SelfAttention("layers.0", get_rope(128, max_position=4096)))
    _bind_kv_cache(model.layers[0].attn)
    specs = init_reanchor_rotary_specs(
        model,
        _kv_cache_config(SlidingWindowSpec(sliding_window=64, **_SPEC_ARGS)),
        default_vllm_config.compilation_config.static_forward_context,
    )
    assert set(specs) == {"layers.0.attn"}


@pytest.mark.parametrize(
    "spec",
    [
        FullAttentionSpec(**_SPEC_ARGS),
        # With the hybrid KV cache manager off, a sliding window layer is
        # allocated as full attention.
        FullAttentionSpec(sliding_window=64, **_SPEC_ARGS),
        SlidingWindowMLASpec(sliding_window=64, **_SPEC_ARGS),
    ],
    ids=["full", "full-with-window", "mla"],
)
def test_init_reanchor_rotary_specs_requires_sliding_window(default_vllm_config, spec):
    model = _model(_SelfAttention("layers.0", get_rope(128, max_position=4096)))
    with pytest.raises(ValueError, match="sliding window"):
        init_reanchor_rotary_specs(
            model,
            _kv_cache_config(spec),
            default_vllm_config.compilation_config.static_forward_context,
        )


@pytest.mark.parametrize(
    "spec_args, num_blocks",
    [
        (dict(kv_quant_mode=KVQuantMode.FP8_PER_TENSOR), 8),
        (dict(num_head_slots=2), 8),
        # Kernel blocks: the block table's ids do not index this view.
        (dict(), 16),
    ],
    ids=["quantized", "head-slots", "kernel-blocks"],
)
def test_init_reanchor_rotary_specs_rejects_kv_layout(
    default_vllm_config, spec_args, num_blocks
):
    model = _model(_SelfAttention("layers.0", get_rope(128, max_position=4096)))
    _bind_kv_cache(model.layers[0].attn, num_blocks)
    spec = SlidingWindowSpec(sliding_window=64, **_SPEC_ARGS, **spec_args)
    with pytest.raises(ValueError, match="KV cache of"):
        init_reanchor_rotary_specs(
            model,
            _kv_cache_config(spec),
            default_vllm_config.compilation_config.static_forward_context,
        )


def test_init_reanchor_rotary_specs_rejects_kv_sharing(default_vllm_config):
    """A layer sharing another layer's KV would rotate the same keys twice."""
    rope = get_rope(128, max_position=4096)
    model = _model(_SelfAttention("layers.0", rope))
    _bind_kv_cache(model.layers[0].attn)
    shared = _attention("layers.1.attn", kv_sharing_target_layer_name="layers.0.attn")
    _bind_kv_cache(shared)
    spec = SlidingWindowSpec(sliding_window=64, **_SPEC_ARGS)
    with pytest.raises(ValueError, match="unshared"):
        init_reanchor_rotary_specs(
            model,
            _kv_cache_config(spec, ("layers.0.attn", "layers.1.attn")),
            default_vllm_config.compilation_config.static_forward_context,
        )
