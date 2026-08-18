# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DynamicNTKScalingRotaryEmbedding.

Regression tests for a bug where serving a model below its trained
context length drove the Dynamic NTK base negative, producing a complex
RoPE cache with huge magnitudes that silently discarded its imaginary
part when cast to a real dtype (see GH issue about NomicBert NaN
embeddings at max_model_len < max_trained_positions).
"""

import torch

from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_scaling_rope import (
    DynamicNTKScalingRotaryEmbedding,
)


def _make_rope(max_position_embeddings: int, max_trained_positions: int = 2048):
    return DynamicNTKScalingRotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=max_position_embeddings,
        max_trained_positions=max_trained_positions,
        base=1000.0,
        is_neox_style=True,
        scaling_factor=2.0,
        dtype=torch.float32,
    )


class TestDynamicNTKScalingRotaryEmbeddingBelowTrainedLength:
    def test_served_below_trained_length_is_real_and_finite(self, default_vllm_config):
        rope = _make_rope(max_position_embeddings=512, max_trained_positions=2048)
        cache = rope._compute_cos_sin_cache()
        assert not cache.is_complex()
        assert torch.isfinite(cache).all()
        assert cache.abs().max().item() <= 1.0

    def test_served_below_trained_length_matches_unscaled_base(
        self, default_vllm_config
    ):
        # Below the trained length, NTK scaling should not kick in: the
        # cache should match plain RoPE with the original base.
        rope = _make_rope(max_position_embeddings=512, max_trained_positions=2048)
        plain_inv_freq = rope._compute_inv_freq(rope.base)
        plain_cache = _reference_cache(rope.max_position_embeddings, plain_inv_freq)
        assert torch.allclose(rope._compute_cos_sin_cache(), plain_cache)

    def test_served_at_trained_length_is_real_and_finite(self, default_vllm_config):
        rope = _make_rope(max_position_embeddings=2048, max_trained_positions=2048)
        cache = rope._compute_cos_sin_cache()
        assert not cache.is_complex()
        assert torch.isfinite(cache).all()
        assert cache.abs().max().item() <= 1.0

    def test_served_above_trained_length_still_applies_ntk_scaling(
        self, default_vllm_config
    ):
        # Extension behavior above the trained length must be unchanged:
        # the cache should differ from the plain, unscaled RoPE cache.
        rope = _make_rope(max_position_embeddings=4096, max_trained_positions=2048)
        cache = rope._compute_cos_sin_cache()
        assert not cache.is_complex()
        assert torch.isfinite(cache).all()

        plain_inv_freq = rope._compute_inv_freq(rope.base)
        plain_cache = _reference_cache(rope.max_position_embeddings, plain_inv_freq)
        assert not torch.allclose(cache, plain_cache)


def _reference_cache(max_position_embeddings: int, inv_freq: torch.Tensor):
    t = torch.arange(max_position_embeddings, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)
