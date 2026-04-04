"""Tests for LoRAModel, focusing on indexed module name handling.

Covers the fix for PEFT checkpoints where modules inside nn.Sequential or
nn.ModuleList are targeted by numeric index (e.g., ``to_out.0``).
"""

import pytest
import torch

from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.utils import parse_fine_tuned_lora_name


# ---------------------------------------------------------------------------
# _get_effective_module_suffix
# ---------------------------------------------------------------------------

class TestGetEffectiveModuleSuffix:
    """Tests for LoRAModel._get_effective_module_suffix."""

    @pytest.mark.parametrize(
        "module_name, expected_suffix",
        [
            # Normal modules – last segment is the actual module name
            ("transformer_blocks.0.attn.to_k", "to_k"),
            ("model.layers.0.self_attn.q_proj", "q_proj"),
            ("model.layers.0.mlp.gate_proj", "gate_proj"),
            # Indexed modules – trailing digit should be skipped
            ("transformer_blocks.0.attn.to_out.0", "to_out"),
            ("model.blocks.1.ff.net.0", "net"),
            ("encoder.layers.3.projection.0", "projection"),
            # Single-segment names
            ("q_proj", "q_proj"),
            # Edge: single digit name – len(parts) == 1, keep as-is
            ("0", "0"),
            # Edge: two-segment with digit
            ("to_out.0", "to_out"),
            # Alphanumeric suffix (not purely digit) – keep as-is
            ("model.layers.0.attn.attention2", "attention2"),
        ],
    )
    def test_suffix_extraction(self, module_name: str, expected_suffix: str):
        assert LoRAModel._get_effective_module_suffix(module_name) == expected_suffix


# ---------------------------------------------------------------------------
# get_lora_by_indexed_name
# ---------------------------------------------------------------------------

class TestGetLoraByIndexedName:
    """Tests for LoRAModel.get_lora_by_indexed_name."""

    @staticmethod
    def _make_lora_model(lora_keys: list[str]) -> LoRAModel:
        """Create a minimal LoRAModel with dummy weights for given keys."""
        loras: dict[str, LoRALayerWeights] = {}
        for key in lora_keys:
            loras[key] = LoRALayerWeights(
                module_name=key,
                rank=8,
                lora_alpha=16,
                lora_a=torch.zeros(8, 16),
                lora_b=torch.zeros(16, 8),
            )
        return LoRAModel(lora_model_id=1, rank=8, loras=loras)

    def test_exact_match_takes_priority(self):
        """get_lora should find exact name; indexed fallback is not needed."""
        model = self._make_lora_model(["blocks.0.attn.to_out"])
        assert model.get_lora("blocks.0.attn.to_out") is not None
        # No indexed variant exists → fallback returns None
        assert model.get_lora_by_indexed_name("blocks.0.attn.to_out") is None

    def test_indexed_fallback_finds_numeric_suffix(self):
        """When LoRA is stored under 'to_out.0', querying 'to_out' should
        find it via the indexed-name fallback."""
        model = self._make_lora_model([
            "blocks.0.attn.to_k",
            "blocks.0.attn.to_out.0",
        ])
        # Exact match fails
        assert model.get_lora("blocks.0.attn.to_out") is None
        # Indexed fallback succeeds
        result = model.get_lora_by_indexed_name("blocks.0.attn.to_out")
        assert result is not None
        assert result.module_name == "blocks.0.attn.to_out.0"

    def test_no_false_positive_on_similar_prefix(self):
        """'to_out_proj' should NOT match when querying 'to_out'."""
        model = self._make_lora_model(["blocks.0.attn.to_out_proj"])
        assert model.get_lora_by_indexed_name("blocks.0.attn.to_out") is None

    def test_non_digit_suffix_not_matched(self):
        """Trailing non-digit segments should not match."""
        model = self._make_lora_model(["blocks.0.attn.to_out.bias"])
        assert model.get_lora_by_indexed_name("blocks.0.attn.to_out") is None

    def test_multiple_indexed_returns_first(self):
        """If multiple indexed variants exist, one of them is returned."""
        model = self._make_lora_model([
            "blocks.0.attn.to_out.0",
            "blocks.0.attn.to_out.1",
        ])
        result = model.get_lora_by_indexed_name("blocks.0.attn.to_out")
        assert result is not None
        assert result.module_name in (
            "blocks.0.attn.to_out.0",
            "blocks.0.attn.to_out.1",
        )


# ---------------------------------------------------------------------------
# check_unexpected_modules (validation)
# ---------------------------------------------------------------------------

class TestCheckUnexpectedModules:
    """Integration tests for validation of indexed module names."""

    @pytest.mark.parametrize(
        "lora_key, expected_suffix",
        [
            (
                "base_model.model.transformer_blocks.0.attn.to_out.0.lora_A.weight",
                "to_out",
            ),
            (
                "base_model.model.transformer_blocks.0.attn.to_k.lora_A.weight",
                "to_k",
            ),
            (
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
                "q_proj",
            ),
        ],
    )
    def test_parse_and_suffix(self, lora_key: str, expected_suffix: str):
        """parse_fine_tuned_lora_name + _get_effective_module_suffix should
        extract the correct suffix for validation."""
        module_name, _ = parse_fine_tuned_lora_name(lora_key, weights_mapper=None)
        suffix = LoRAModel._get_effective_module_suffix(module_name)
        assert suffix == expected_suffix

    def test_indexed_module_passes_validation(self):
        """to_out.0 should be accepted when 'to_out' is an expected module."""
        expected = {"to_k", "to_q", "to_v", "to_out"}
        key = "base_model.model.transformer_blocks.0.attn.to_out.0.lora_A.weight"
        module_name, _ = parse_fine_tuned_lora_name(key, weights_mapper=None)
        suffix = LoRAModel._get_effective_module_suffix(module_name)
        assert suffix in expected

    def test_truly_unexpected_module_rejected(self):
        """A module whose effective suffix is not expected should be
        reported as unexpected."""
        expected = {"to_k", "to_q", "to_v"}
        key = "base_model.model.transformer_blocks.0.attn.to_out.0.lora_A.weight"
        module_name, _ = parse_fine_tuned_lora_name(key, weights_mapper=None)
        suffix = LoRAModel._get_effective_module_suffix(module_name)
        assert suffix not in expected  # "to_out" is NOT in expected

    def test_expert_modules_unaffected(self):
        """Expert module handling should remain unchanged."""
        key = "base_model.model.layers.0.experts.1.w1.lora_A.weight"
        module_name, _ = parse_fine_tuned_lora_name(key, weights_mapper=None)
        assert ".experts" in module_name


# ---------------------------------------------------------------------------
# End-to-end: from_lora_tensors + indexed lookup
# ---------------------------------------------------------------------------

class TestFromLoraTensorsIndexedModules:
    """Verify that LoRA weights loaded from tensors with indexed module
    names can be retrieved via the indexed-name fallback."""

    @staticmethod
    def _make_dummy_tensors(module_paths: list[str]) -> dict[str, torch.Tensor]:
        """Create fake safetensors-style weight dict."""
        tensors: dict[str, torch.Tensor] = {}
        for path in module_paths:
            tensors[f"base_model.model.{path}.lora_A.weight"] = torch.randn(8, 64)
            tensors[f"base_model.model.{path}.lora_B.weight"] = torch.randn(64, 8)
        return tensors

    def test_indexed_module_loaded_and_retrievable(self):
        """Weights for 'to_out.0' should be loadable and findable by
        querying 'to_out' through the indexed fallback."""
        from vllm.lora.peft_helper import PEFTHelper

        peft_helper = PEFTHelper(
            r=8,
            lora_alpha=16,
            target_modules=["to_k", "to_q", "to_v", "to_out"],
        )

        tensors = self._make_dummy_tensors([
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_out.0",
        ])

        lora_model = LoRAModel.from_lora_tensors(
            lora_model_id=1,
            tensors=tensors,
            peft_helper=peft_helper,
            device="cpu",
        )

        # Exact name lookup for normal module
        assert lora_model.get_lora("transformer_blocks.0.attn.to_k") is not None

        # Exact name lookup for indexed module – stored under original key
        assert lora_model.get_lora("transformer_blocks.0.attn.to_out.0") is not None

        # Indexed fallback – querying without the ".0" suffix
        result = lora_model.get_lora_by_indexed_name(
            "transformer_blocks.0.attn.to_out"
        )
        assert result is not None
        assert result.lora_a is not None
        assert result.lora_b is not None
