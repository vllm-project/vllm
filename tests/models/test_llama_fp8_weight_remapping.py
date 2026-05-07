# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP8 weight name remapping in LlamaForCausalLM.

FP8 quantized HuggingFace checkpoints (e.g. Devstral-2-123B-Instruct)
store per-tensor scales as ``activation_scale`` and ``weight_scale_inv``,
but vLLM's FP8 linear layers register them as ``input_scale`` and
``weight_scale``.  The remapping added to ``LlamaForCausalLM`` and
``LlamaModel`` ensures these names are translated during weight loading.
"""

import pytest

from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.utils import WeightsMapper


# ---------------------------------------------------------------------------
# WeightsMapper suffix remapping (unit test, no GPU required)
# ---------------------------------------------------------------------------

@pytest.mark.cpu_test
class TestLlamaFP8WeightsMapper:
    """Verify the hf_to_vllm_mapper on LlamaForCausalLM correctly
    remaps FP8 scale names used by HuggingFace checkpoints."""

    @pytest.fixture
    def mapper(self) -> WeightsMapper:
        return LlamaForCausalLM.hf_to_vllm_mapper

    def test_activation_scale_remapped(self, mapper: WeightsMapper):
        result = mapper._map_name(
            "model.layers.0.mlp.down_proj.activation_scale"
        )
        assert result == "model.layers.0.mlp.down_proj.input_scale"

    def test_weight_scale_inv_remapped(self, mapper: WeightsMapper):
        result = mapper._map_name(
            "model.layers.0.mlp.down_proj.weight_scale_inv"
        )
        assert result == "model.layers.0.mlp.down_proj.weight_scale"

    def test_non_fp8_name_unchanged(self, mapper: WeightsMapper):
        name = "model.layers.0.self_attn.q_proj.weight"
        assert mapper._map_name(name) == name

    def test_all_linear_suffixes(self, mapper: WeightsMapper):
        """Ensure both suffixes are remapped for every linear layer type."""
        linear_names = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ]
        for prefix in linear_names:
            assert mapper._map_name(f"{prefix}.activation_scale") == \
                f"{prefix}.input_scale", f"Failed for {prefix}.activation_scale"
            assert mapper._map_name(f"{prefix}.weight_scale_inv") == \
                f"{prefix}.weight_scale", f"Failed for {prefix}.weight_scale_inv"

    def test_mapper_as_generator(self, mapper: WeightsMapper):
        """Verify apply() works as a lazy generator over weight tuples."""
        import torch
        fake_weights = [
            ("model.layers.0.mlp.down_proj.activation_scale", torch.tensor(1.0)),
            ("model.layers.0.mlp.down_proj.weight_scale_inv", torch.tensor(0.5)),
            ("model.layers.0.mlp.down_proj.weight", torch.randn(4, 4)),
        ]
        remapped = dict(mapper.apply(fake_weights))
        assert "model.layers.0.mlp.down_proj.input_scale" in remapped
        assert "model.layers.0.mlp.down_proj.weight_scale" in remapped
        assert "model.layers.0.mlp.down_proj.weight" in remapped
        assert "model.layers.0.mlp.down_proj.activation_scale" not in remapped
        assert "model.layers.0.mlp.down_proj.weight_scale_inv" not in remapped
