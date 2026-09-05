# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from transformers import AutoConfig

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs import SolarOpen2Config

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def solar_open2_config_dict():
    return {
        "model_type": "solar_open2",
        "architectures": ["SolarOpen2ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 48,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 1_048_576,
        "rope_theta": 10000,
        "gqa_interval": 3,
        "gqa_layers": list(range(0, 48, 4)),
        "n_routed_experts": 320,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "linear_attn_config": {
            "short_conv_kernel_size": 4,
            "head_dim": 128,
            "num_heads": 64,
            "num_kv_heads": None,
        },
    }


def test_get_config_without_remote_code(tmp_path, solar_open2_config_dict):
    (tmp_path / "config.json").write_text(json.dumps(solar_open2_config_dict))

    config = get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(config, SolarOpen2Config)
    assert config.gqa_layers == list(range(0, 48, 4))
    assert config.layer_types.count("full_attention") == 12
    assert config.layer_types[0] == "full_attention"
    assert config.layer_types[1] == "linear_attention"
    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 10000,
        "partial_rotary_factor": 1.0,
    }

    assert isinstance(AutoConfig.from_pretrained(tmp_path), SolarOpen2Config)


def test_gqa_interval_derives_layer_types():
    """gqa_interval=N: one full-attention layer then N linear-attention
    layers, starting at layer 0 (matches the Transformers implementation)."""
    expected = [
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
    ]

    config = SolarOpen2Config(num_hidden_layers=6, gqa_layers=[0, 4])
    assert config.layer_types == expected

    config = SolarOpen2Config(num_hidden_layers=6, gqa_interval=3)
    assert config.gqa_layers is None
    assert config.layer_types == expected

    config = SolarOpen2Config(num_hidden_layers=6, gqa_interval=5)
    assert config.layer_types.count("full_attention") == 1
    assert config.layer_types[0] == "full_attention"

    # When both are given, gqa_layers wins over gqa_interval.
    config = SolarOpen2Config(num_hidden_layers=6, gqa_layers=[1], gqa_interval=3)
    assert config.layer_types.count("full_attention") == 1
    assert config.layer_types[1] == "full_attention"


def test_layer_types_priority_over_gqa_layers():
    """Explicit layer_types wins over gqa_layers / gqa_interval."""
    config = SolarOpen2Config(
        num_hidden_layers=4,
        gqa_layers=[0],
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )

    assert config.layer_types.count("full_attention") == 1
    assert config.layer_types[3] == "full_attention"


def test_rejects_invalid_layer_types():
    with pytest.raises(ValueError, match="one valid attention type per layer"):
        SolarOpen2Config(num_hidden_layers=2, layer_types=["full_attention"])


def test_rejects_invalid_gqa_interval():
    with pytest.raises(ValueError, match="positive integer"):
        SolarOpen2Config(num_hidden_layers=4, gqa_interval=0)


def test_rejects_non_integral_gqa_interval():
    with pytest.raises(ValueError, match="positive integer"):
        SolarOpen2Config(num_hidden_layers=4, gqa_interval=1.5)


def test_rejects_non_integral_gqa_layers():
    with pytest.raises(ValueError, match="integer layer indices"):
        SolarOpen2Config(num_hidden_layers=4, gqa_layers=[0.0, 1.5])


def test_rejects_bool_gqa_interval():
    with pytest.raises(ValueError, match="positive integer"):
        SolarOpen2Config(num_hidden_layers=4, gqa_interval=True)


def test_rejects_bool_gqa_layers():
    with pytest.raises(ValueError, match="integer layer indices"):
        SolarOpen2Config(num_hidden_layers=4, gqa_layers=[True, 2])


def test_rejects_out_of_range_gqa_layers():
    with pytest.raises(ValueError, match="integer layer indices"):
        SolarOpen2Config(num_hidden_layers=4, gqa_layers=[0, 7])


def test_rejects_all_linear_pattern():
    with pytest.raises(ValueError, match="at least one full-attention"):
        SolarOpen2Config(num_hidden_layers=4, gqa_layers=[])


def test_rejects_incomplete_linear_attn_config():
    with pytest.raises(ValueError, match="missing required keys"):
        SolarOpen2Config(linear_attn_config={"num_heads": 64})


def test_rejects_unsupported_kda_num_kv_heads():
    with pytest.raises(ValueError, match="num_kv_heads"):
        SolarOpen2Config(
            linear_attn_config={
                "short_conv_kernel_size": 4,
                "head_dim": 128,
                "num_heads": 64,
                "num_kv_heads": 8,
            }
        )


def test_rope_parameters_get_default_partial_rotary_factor():
    """partial_rotary_factor defaults to 1.0 even for explicit rope_parameters
    (matches SolarOpen2Config in Transformers)."""
    config = SolarOpen2Config(
        rope_parameters={
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 65536,
        }
    )

    assert config.rope_parameters["partial_rotary_factor"] == 1.0


def test_rope_theta_merged_into_explicit_rope_parameters():
    """A top-level rope_theta must reach rope_parameters, which is what
    get_rope() reads, even when rope_parameters is given explicitly."""
    config = SolarOpen2Config(
        rope_theta=500000.0,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 65536,
        },
    )

    assert config.rope_parameters["rope_theta"] == 500000.0
    # rope_parameters is the single source of truth for rope settings.
    assert "rope_theta" not in config.to_dict()


def test_rope_parameters_theta_wins_over_top_level_rope_theta():
    config = SolarOpen2Config(
        rope_theta=500000.0,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )

    assert config.rope_parameters["rope_theta"] == 1000000.0


def test_legacy_rope_scaling_and_rope_theta():
    """A pre-v5 config carries the old `rope_scaling`/`type` keys next to a
    top-level rope_theta; all of it has to end up in rope_parameters."""
    config = SolarOpen2Config(
        max_position_embeddings=262144,
        rope_theta=250000.0,
        rope_scaling={"type": "yarn", "factor": 4.0},
    )

    assert config.rope_parameters["rope_type"] == "yarn"
    assert config.rope_parameters["factor"] == 4.0
    assert config.rope_parameters["rope_theta"] == 250000.0
    assert "type" not in config.rope_parameters
    # Seeded by Transformers because rope_parameters is set before
    # super().__init__(); get_rope() indexes it unguarded for yarn.
    assert config.rope_parameters["original_max_position_embeddings"] == 262144
    assert "rope_theta" not in config.to_dict()


def test_partial_rotary_factor_is_not_left_standalone():
    config = SolarOpen2Config(partial_rotary_factor=0.5)

    assert config.rope_parameters["partial_rotary_factor"] == 0.5
    assert "partial_rotary_factor" not in config.to_dict()


def test_rejects_incomplete_scaled_rope_parameters():
    """Setting rope_parameters before super().__init__() lets Transformers
    reject a scaled rope shape that get_rope() could not consume."""
    with pytest.raises(KeyError, match="factor"):
        SolarOpen2Config(rope_parameters={"rope_type": "yarn"})
