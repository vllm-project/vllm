# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Fp8Config's handling of whole modules in `modules_to_not_convert`.

Regression test: HF FP8 exports of Ministral-3 and Devstral-Small-2 keep the
vision tower and projector in BF16 and list them by module name
(`model.vision_tower`). The names were mapped bare, so `model.vision_tower` fell
through Mistral3's `model.vision_tower.` prefix rule to the catch-all `model.`,
and the exact match never hit the layers inside the module. Served with
`--config-format hf`, those layers were built as FP8.
"""

import pytest

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models.mistral3 import Mistral3ForConditionalGeneration

# The real list from Ministral-3-14B-Instruct-2512's quantization_config.
MINISTRAL3_IGNORE = ["model.vision_tower", "model.multi_modal_projector", "lm_head"]


def _config(ignored_layers: list[str]) -> Fp8Config:
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="static",
        ignored_layers=ignored_layers,
    )
    config.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    return config


def test_module_names_map_through_trailing_dot_prefix_rules():
    config = _config(MINISTRAL3_IGNORE)
    config.apply_vllm_mapper(Mistral3ForConditionalGeneration.hf_to_vllm_mapper)
    assert {"vision_tower", "multi_modal_projector", "language_model.lm_head"} <= set(
        config.ignored_layers
    )


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("vision_tower.transformer.layers.0.attention.qkv_proj", True),
        ("vision_tower.transformer.layers.0.feed_forward.down_proj", True),
        ("multi_modal_projector.linear_1", True),
        ("multi_modal_projector.patch_merger.merging_layer", True),
        ("language_model.lm_head", True),
        ("language_model.model.layers.0.self_attn.qkv_proj", False),
        ("language_model.model.layers.0.mlp.gate_up_proj", False),
        # A name that only shares a prefix with an ignored module.
        ("vision_tower_extra.linear", False),
    ],
)
def test_layers_inside_ignored_modules_stay_unquantized(prefix, expected):
    config = _config(MINISTRAL3_IGNORE)
    config.apply_vllm_mapper(Mistral3ForConditionalGeneration.hf_to_vllm_mapper)
    assert config.is_ignored(prefix) is expected


def test_full_layer_names_still_match():
    """Entries naming single layers, and fused shards, behave as before."""
    config = _config(
        [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.1.mlp.down_proj",
        ]
    )
    assert config.is_ignored("model.layers.0.self_attn.qkv_proj")
    assert config.is_ignored("model.layers.1.mlp.down_proj")
    assert not config.is_ignored("model.layers.1.mlp.gate_up_proj")
    assert not config.is_ignored("model.layers.10.mlp.down_proj")
