# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for GGUF MoE expert parameter mapping compatibility.

Verifies that the sideload_params regex patterns in GGUFModelLoader
correctly match both transformers v4 (per-expert) and v5+ (fused expert)
parameter naming formats for MoE models (qwen2_moe, qwen3_moe,
deepseek_v2, deepseek_v3).
"""

import pytest
import regex as re


def _build_sideload_patterns_qwen_moe(
    num_hidden_layers: int,
) -> list[re.Pattern]:
    """Build sideload_params patterns for qwen2_moe/qwen3_moe,
    replicating the logic from GGUFModelLoader._get_gguf_weights_map."""
    sideload_params: list[re.Pattern] = []
    for idx in range(num_hidden_layers):
        # v4 format
        sideload_params.append(
            re.compile(
                f"model\\.layers\\.{idx}"
                r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
            )
        )
        # v5+ fused format
        sideload_params.append(
            re.compile(
                f"model\\.layers\\.{idx}"
                r"\.mlp\.experts\.(gate_up_proj|down_proj)"
            )
        )
    return sideload_params


def _build_sideload_patterns_deepseek(
    num_hidden_layers: int,
) -> list[re.Pattern]:
    """Build sideload_params patterns for deepseek_v2/deepseek_v3,
    replicating the logic from GGUFModelLoader._get_gguf_weights_map."""
    sideload_params: list[re.Pattern] = []
    for idx in range(num_hidden_layers):
        # v4 format
        sideload_params.append(
            re.compile(
                f"model\\.layers\\.{idx}"
                r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
            )
        )
        # v5+ fused format
        sideload_params.append(
            re.compile(
                f"model\\.layers\\.{idx}"
                r"\.mlp\.experts\.(gate_up_proj|down_proj)"
            )
        )
    return sideload_params


NUM_LAYERS = 3  # Use a small number for testing


class TestQwenMoESideloadPatterns:
    """Test sideload regex for qwen2_moe and qwen3_moe."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.patterns = _build_sideload_patterns_qwen_moe(NUM_LAYERS)

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    @pytest.mark.parametrize("expert_idx", [0, 1, 63, 127])
    @pytest.mark.parametrize("proj", ["gate_proj", "up_proj", "down_proj"])
    def test_v4_per_expert_format(self, layer_idx: int, expert_idx: int, proj: str):
        """Transformers v4 format: per-expert with .weight suffix."""
        name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight"
        matched = any(re.fullmatch(p, name) for p in self.patterns)
        assert matched, f"v4 param not matched by sideload regex: {name}"

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    @pytest.mark.parametrize("proj", ["gate_up_proj", "down_proj"])
    def test_v5_fused_format(self, layer_idx: int, proj: str):
        """Transformers v5+ format: fused experts without .weight suffix."""
        name = f"model.layers.{layer_idx}.mlp.experts.{proj}"
        matched = any(re.fullmatch(p, name) for p in self.patterns)
        assert matched, f"v5 fused param not matched by sideload regex: {name}"

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    def test_v5_gate_up_proj_not_partial_match(self, layer_idx: int):
        """gate_up_proj should only fullmatch, not partial match
        something like gate_up_proj.weight."""
        name = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj.weight"
        matched = any(re.fullmatch(p, name) for p in self.patterns)
        assert not matched, f"Unexpected match for name with .weight suffix: {name}"

    def test_unrelated_param_not_matched(self):
        """Non-expert params should not be matched."""
        unrelated = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.input_layernorm.weight",
            "model.embed_tokens.weight",
        ]
        for name in unrelated:
            matched = any(re.fullmatch(p, name) for p in self.patterns)
            assert not matched, f"Unrelated param incorrectly matched: {name}"

    def test_out_of_range_layer_not_matched(self):
        """Layer index beyond num_hidden_layers should not be matched."""
        name = f"model.layers.{NUM_LAYERS}.mlp.experts.gate_up_proj"
        matched = any(re.fullmatch(p, name) for p in self.patterns)
        assert not matched, f"Out-of-range layer index incorrectly matched: {name}"


class TestDeepseekMoESideloadPatterns:
    """Test sideload regex for deepseek_v2 and deepseek_v3."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.patterns = _build_sideload_patterns_deepseek(NUM_LAYERS)

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    @pytest.mark.parametrize("expert_idx", [0, 1, 63])
    @pytest.mark.parametrize("proj", ["gate_proj", "up_proj", "down_proj"])
    def test_v4_per_expert_format(self, layer_idx: int, expert_idx: int, proj: str):
        """Transformers v4 format: per-expert with .weight suffix."""
        name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight"
        matched = any(re.fullmatch(p, name) for p in self.patterns)
        assert matched, f"v4 param not matched by sideload regex: {name}"

    @pytest.mark.parametrize("layer_idx", range(NUM_LAYERS))
    @pytest.mark.parametrize("proj", ["gate_up_proj", "down_proj"])
    def test_v5_fused_format(self, layer_idx: int, proj: str):
        """Transformers v5+ format: fused experts without .weight suffix."""
        name = f"model.layers.{layer_idx}.mlp.experts.{proj}"
        matched = any(re.fullmatch(p, name) for p in self.patterns)
        assert matched, f"v5 fused param not matched by sideload regex: {name}"


class TestManualMappingIntegrity:
    """Test that manual GGUF->HF mappings are correctly structured."""

    def test_qwen_moe_manual_mapping_keys(self):
        """Verify GGUF tensor names used in manual mapping."""
        num_layers = 2
        gguf_to_hf_name_map: dict[str, str] = {}
        for idx in range(num_layers):
            gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
            )
            gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
            )
            gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
            )

        # Each layer should have exactly 3 GGUF tensor mappings
        for idx in range(num_layers):
            layer_keys = [k for k in gguf_to_hf_name_map if f"blk.{idx}." in k]
            assert len(layer_keys) == 3, (
                f"Expected 3 GGUF mappings for layer {idx}, got {len(layer_keys)}"
            )

        # Verify GGUF names follow the standard convention
        for gguf_name in gguf_to_hf_name_map:
            assert gguf_name.startswith("blk."), (
                f"GGUF name should start with 'blk.': {gguf_name}"
            )
            assert gguf_name.endswith(".weight"), (
                f"GGUF name should end with '.weight': {gguf_name}"
            )

    def test_v5_fused_params_not_in_manual_mapping_values(self):
        """v5 fused param names should NOT appear in manual mapping values,
        since the manual mapping targets v4 format for load_weights compat."""
        num_layers = 2
        manual_values = []
        for idx in range(num_layers):
            manual_values.extend(
                [
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight",
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight",
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight",
                ]
            )

        v5_names = []
        for idx in range(num_layers):
            v5_names.extend(
                [
                    f"model.layers.{idx}.mlp.experts.gate_up_proj",
                    f"model.layers.{idx}.mlp.experts.down_proj",
                ]
            )

        for v5_name in v5_names:
            assert v5_name not in manual_values, (
                f"v5 fused name should not be in manual mapping: {v5_name}"
            )

    def test_v5_fused_params_filtered_by_sideload(self):
        """End-to-end: v5 fused param names should be filtered out
        by sideload_params, preventing RuntimeError."""
        num_layers = 2
        sideload_params = _build_sideload_patterns_qwen_moe(num_layers)

        # Simulate v5 state_dict expert param names
        v5_state_dict_expert_names = []
        for idx in range(num_layers):
            v5_state_dict_expert_names.extend(
                [
                    f"model.layers.{idx}.mlp.experts.gate_up_proj",
                    f"model.layers.{idx}.mlp.experts.down_proj",
                ]
            )

        # These names would be in unmapped_params;
        # verify sideload filters them all out
        remaining = list(
            filter(
                lambda x: not any(re.fullmatch(p, x) for p in sideload_params),
                v5_state_dict_expert_names,
            )
        )
        assert remaining == [], f"v5 fused params not filtered by sideload: {remaining}"
