# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.lora.utils import is_in_target_modules


class TestIsInTargetModules:
    """Tests for is_in_target_modules (deployment-time filter)."""

    def test_none_allows_all(self):
        assert is_in_target_modules("model.layers.0.self_attn.o_proj", None)

    def test_suffix_in_target(self):
        assert is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["o_proj", "q_proj"]
        )

    def test_suffix_not_in_target(self):
        assert not is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["q_proj", "k_proj"]
        )

    def test_empty_target_modules(self):
        assert not is_in_target_modules("model.layers.0.self_attn.o_proj", [])

    def test_exact_name_match(self):
        assert is_in_target_modules("dense1", ["dense1", "dense2"])

    def test_exact_name_no_match(self):
        assert not is_in_target_modules("dense3", ["dense1", "dense2"])

    def test_packed_parent_matches_child_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.gate_up_proj",
            ["gate_proj", "up_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )

    def test_packed_child_matches_parent_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.gate_proj",
            ["gate_up_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )

    def test_packed_child_matches_full_parent_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.gate_proj",
            ["model.layers.0.mlp.gate_up_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )

    def test_dotted_moe_child_matches_parent_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.experts.0.gate_proj",
            ["experts"],
            {"experts": ["experts.0.gate_proj", "experts.0.up_proj"]},
        )

    def test_dotted_moe_child_leaf_load_apply_consistency(self):
        packed_modules_mapping = {
            "experts": ["experts.0.gate_proj", "experts.0.up_proj"]
        }
        target_modules = ["gate_proj"]

        assert is_in_target_modules(
            "model.layers.0.mlp.experts.0.gate_proj",
            target_modules,
            packed_modules_mapping,
        )
        assert is_in_target_modules(
            "model.layers.0.mlp.experts",
            target_modules,
            packed_modules_mapping,
        )

    def test_runtime_prefix_missing_from_adapter_module_name(self):
        assert is_in_target_modules(
            "foo.q_proj",
            ["model.foo.q_proj"],
            module_name_prefix="model.",
        )

    def test_runtime_prefix_missing_from_target_module_name(self):
        assert is_in_target_modules(
            "model.foo.q_proj",
            ["foo.q_proj"],
            module_name_prefix="model.",
        )

    def test_fused_parent_matches_child_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.self_attn.fused_qkv_a_proj",
            ["q_a_proj", "kv_a_proj_with_mqa"],
            {"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]},
        )
