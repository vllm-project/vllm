# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for humming's is_layer_skipped handling of compressed-tensors
regex ("re:") ignore entries.

Regression test: compressed-tensors checkpoints (e.g. Kimi-K2.6) list ignored
layers as regex patterns prefixed with "re:" (e.g. "re:vision_tower.*").
humming previously substring-matched these literals, so ignored layers were
never skipped and got (incorrectly) quantized -- which then diverged between
the real weight-load path (falls back to unquantized) and the dummy-load path
(stays quantized), breaking any consumer that byte-compares the two.
"""

import pytest

from vllm.model_executor.layers.quantization.humming import HummingConfig

# The real ignore list from Kimi-K2.6's compressed-tensors quantization_config.
K26_IGNORE = [
    "re:.*self_attn.*",
    "re:.*shared_experts.*",
    r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
    "re:.*lm_head.*",
    "re:vision_tower.*",
    "re:mm_projector.*",
]


@pytest.mark.parametrize(
    "prefix,expected",
    [
        # ignored layers (must be skipped -> stay unquantized)
        ("vision_tower.encoder.blocks.0.mlp.fc0", True),
        ("vision_tower.encoder.blocks.16.wo", True),
        ("mm_projector.linear_1", True),
        ("language_model.model.layers.0.self_attn.o_proj", True),
        ("language_model.model.layers.0.mlp.gate_up_proj", True),
        ("language_model.model.layers.0.mlp.down_proj", True),
        ("language_model.model.layers.3.mlp.shared_experts.gate_proj", True),
        ("model.lm_head", True),
        # routed experts are NOT ignored -> must be quantized
        ("language_model.model.layers.3.mlp.experts.383.up_proj", False),
        ("language_model.model.layers.3.mlp.experts.0.down_proj", False),
    ],
)
def test_is_layer_skipped_regex_ignore(prefix, expected):
    cfg = HummingConfig(full_config={"ignore": K26_IGNORE})
    assert cfg.is_layer_skipped({"ignore": K26_IGNORE}, prefix) is expected


def test_plain_substring_entries_still_work():
    # bitsandbytes-style modules_to_not_convert use plain substrings.
    cfg = HummingConfig()
    cfg_dict = {"modules_to_not_convert": ["lm_head", "vision"]}
    assert cfg.is_layer_skipped(cfg_dict, "model.vision.encoder.fc") is True
    assert cfg.is_layer_skipped(cfg_dict, "model.layers.0.mlp.up_proj") is False
