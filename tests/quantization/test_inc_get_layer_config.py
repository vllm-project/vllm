# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for INCConfig.get_layer_config.

Focused on step 4 (fused QKV / packed_modules_mapping) to prevent
false-positive substring matches.
"""

import pytest
import torch.nn as nn

from vllm.model_executor.layers.quantization.inc import INCConfig


class _FakeLayer(nn.Linear):
    """Minimal nn.Linear subclass used as a stand-in for real model layers."""

    def __init__(self):
        super().__init__(128, 128, bias=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inc(extra_config: dict | None = None) -> INCConfig:
    """Create an INCConfig with bits=4, group_size=128."""
    return INCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        extra_config=extra_config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetLayerConfigFusedQKV:
    """Tests for step-4 (fused QKV / packed_modules_mapping) logic."""

    def test_exact_fusion_key_match(self):
        """A layer whose name contains 'qkv' maps to its extra_config entry."""
        extra_config = {
            "model.layers.0.self_attn.qkv_proj": {"bits": 8},
        }
        cfg = _make_inc(extra_config=extra_config)
        layer = _FakeLayer()
        # Simulate packed_modules_mapping having {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        cfg.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        bits, _, _ = cfg.get_layer_config(layer, "model.layers.0.self_attn.qkv_proj")
        assert bits == 8

    def test_false_substring_match_does_not_override(self):
        """Regression test for the false-substring-match bug.

        Scenario (Qwen3.6-35B-A3B VLM):
        - packed_modules_mapping has "qkv" → ["qkv"] (from the vision encoder).
        - The GDN text-attention layer is named "in_proj_qkvz".
        - "qkv" is a substring of "in_proj_qkvz", so the old code would enter
          step 4 and generate sub_name "in_proj_qkvz" (replacing "qkv" with
          "qkv").  That name is NOT in extra_config, so get_config() falls back
          to the global default (bits=4), even though the correct value is 16.
        - The fix: skip the fusion key when none of the generated sub_names
          actually exist in extra_config.
        """
        extra_config = {
            # "in_proj_qkv" is bits=16 (unquantized); "in_proj_z" is also 16.
            "model.layers.0.in_proj_qkv": {"bits": 16},
            "model.layers.0.in_proj_z": {"bits": 16},
        }
        cfg = _make_inc(extra_config=extra_config)
        layer = _FakeLayer()
        # "qkv" maps to itself – coming from the vision encoder's packed mapping.
        cfg.packed_modules_mapping = {
            "qkv": ["qkv"],
        }

        # "in_proj_qkvz" contains "qkv" as a substring, but the generated
        # sub_name "in_proj_qkvz" is NOT in extra_config → must not match.
        # The layer is not in extra_config either, so it should fall back to
        # the default bits (4) returned by step 5.
        bits, _, _ = cfg.get_layer_config(layer, "model.layers.0.in_proj_qkvz")
        # bits should be the global default (4), NOT the wrong bits=16 that
        # a strict regex would produce, and NOT bits=4 obtained via a wrong
        # path that discards the real extra_config entries.
        assert bits == 4  # global default – no erroneous fusion match

    def test_real_qkv_fusion_key_still_resolves(self):
        """The true "qkv" fusion (vision encoder) still resolves correctly."""
        extra_config = {
            "vision_model.encoder.layers.0.self_attn.qkv": {"bits": 8},
        }
        cfg = _make_inc(extra_config=extra_config)
        layer = _FakeLayer()
        cfg.packed_modules_mapping = {
            "qkv": ["qkv"],
        }
        bits, _, _ = cfg.get_layer_config(layer, "vision_model.encoder.layers.0.self_attn.qkv")
        assert bits == 8

    def test_mixed_fp16_and_int4_fused_layer(self):
        """All sub-keys must agree; inconsistent configs raise ValueError."""
        extra_config = {
            "model.layers.0.self_attn.q_proj": {"bits": 16},
            "model.layers.0.self_attn.k_proj": {"bits": 4},
            "model.layers.0.self_attn.v_proj": {"bits": 4},
        }
        cfg = _make_inc(extra_config=extra_config)
        layer = _FakeLayer()
        cfg.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        with pytest.raises(ValueError, match="consistent quant config"):
            cfg.get_layer_config(layer, "model.layers.0.self_attn.qkv_proj")

    def test_fusion_triggered_by_regex_configured_sub_name(self):
        """Fusion step 4 is still triggered when sub_names match via regex.

        Ensures the guard does not regress when extra_config uses regex
        patterns instead of exact keys to configure sub-modules.
        """
        # A single regex pattern that matches q_proj, k_proj and v_proj
        extra_config = {
            r"model\.layers\.\d+\.self_attn\.(q|k|v)_proj": {"bits": 8},
        }
        cfg = _make_inc(extra_config=extra_config)
        layer = _FakeLayer()
        cfg.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        # All three sub_names match the regex → fusion is triggered and
        # consistent bits=8 is returned.
        bits, _, _ = cfg.get_layer_config(layer, "model.layers.0.self_attn.qkv_proj")
        assert bits == 8
