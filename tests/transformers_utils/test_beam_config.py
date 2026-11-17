# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that Beam's released config schema loads and round-trips."""

import json

import pytest

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.beam import BeamConfig

_NUM_LAYERS = 6
# Shaped like reflection/Beam-1.0-501B-A23B's config.json, scaled down.
_RELEASE_CONFIG = {
    "architectures": ["BeamForCausalLM"],
    "model_type": "beam",
    "attention_bias": False,
    "final_logit_softcapping": 30.0,
    "head_dim": 16,
    "hidden_size": 64,
    "intermediate_size": 128,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    "max_position_embeddings": 1048576,
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 3,
    "moe_intermediate_size": 32,
    "n_group": 8,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "num_hidden_layers": _NUM_LAYERS,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": None,
    },
    "rope_theta": None,
    "routed_scaling_factor": 2.5,
    "sliding_window": 512,
    "tie_word_embeddings": False,
    "topk_group": 4,
    "vocab_size": 256,
}


def test_release_config_loads_and_round_trips(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_RELEASE_CONFIG))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, BeamConfig)
    for resolved in (config, BeamConfig(**config.to_dict())):
        # The NoPE full-attention entry carries no RoPE parameters.
        assert resolved.rope_parameters == {
            "rope_type": "default",
            "rope_theta": 10000.0,
        }
        assert resolved.layer_types == _RELEASE_CONFIG["layer_types"]
        assert resolved.mlp_layer_types == _RELEASE_CONFIG["mlp_layer_types"]
        assert resolved.routed_scaling_factor == 2.5
        assert resolved.final_logit_softcapping == 30.0


def test_mismatched_rope_types_are_rejected():
    rope = {"rope_type": "default", "rope_theta": 10000.0}
    with pytest.raises(ValueError, match="exactly one type"):
        BeamConfig(
            num_hidden_layers=2,
            rope_parameters={"sliding_attention": rope, "full_attention": rope},
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"layer_types": ["sliding_attention"]},
        {"layer_types": ["sliding_attention", "chunked_attention"]},
        {"mlp_layer_types": ["dense"]},
        {"mlp_layer_types": ["dense", "shared"]},
    ],
)
def test_invalid_layer_schedules_are_rejected(overrides):
    with pytest.raises(ValueError, match="one"):
        BeamConfig(num_hidden_layers=2, **overrides)
