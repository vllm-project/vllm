# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.minicpm5_moe import MiniCPM5MoEConfig


def test_checkpoint_names_and_explicit_dimensions_without_remote_code(tmp_path):
    """The 15B checkpoint has head_dim != hidden_size / num_attention_heads."""
    data = {
        "model_type": "minicpm5_moe",
        "architectures": ["MiniCPM5MoEForCausalLM"],
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "num_experts_per_tok": 16,
        "n_routed_experts": 160,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": 3.66,
        "use_gated_attention": True,
        "rope_theta": 5000000,
        "eos_token_id": [1, 130073],
        "auto_map": {"AutoConfig": "must_not_execute.Config"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    (tmp_path / "must_not_execute.py").write_text(
        "raise RuntimeError('remote code executed')"
    )
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, MiniCPM5MoEConfig)
    for key, value in data.items():
        if key != "auto_map":
            assert getattr(config, key) == value, key
    assert config.rope_parameters["rope_theta"] == 5000000
    assert config.rope_parameters["rope_type"] == "default"


@pytest.mark.parametrize("field", ["rope_scaling", "rope_parameters"])
def test_legacy_rope_type_is_normalized(field):
    scaling = {"type": "linear", "factor": 2.0}
    config = MiniCPM5MoEConfig(rope_theta=12345, **{field: scaling})
    assert config.rope_parameters == {
        "rope_type": "linear",
        "factor": 2.0,
        "rope_theta": 12345,
    }
    assert scaling == {"type": "linear", "factor": 2.0}


@pytest.mark.parametrize("include_legacy", [False, True])
def test_explicit_v5_rope_wins_and_round_trips(tmp_path, include_legacy):
    parameters = {"rope_type": "linear", "factor": 3.0, "rope_theta": 45678}
    scaling = {"type": "linear"}
    legacy = {"rope_scaling": scaling} if include_legacy else {}
    config = MiniCPM5MoEConfig(rope_theta=12345, rope_parameters=parameters, **legacy)
    assert config.rope_parameters == parameters
    assert config.rope_theta == 45678
    config.save_pretrained(tmp_path)
    reloaded = MiniCPM5MoEConfig.from_pretrained(tmp_path)
    assert reloaded.rope_parameters == parameters
    assert reloaded.rope_theta == 45678
    assert parameters == {"rope_type": "linear", "factor": 3.0, "rope_theta": 45678}
    assert scaling == {"type": "linear"}


def test_checkpoint_defaults():
    config = MiniCPM5MoEConfig()
    assert config.head_dim == 128
    assert config.num_key_value_heads == 2
    assert config.num_experts_per_tok == 16
    assert config.eos_token_id == [1, 130073]


def test_explicit_none_preserves_inference_and_forwarded_kwargs():
    config = MiniCPM5MoEConfig(
        hidden_size=96,
        num_attention_heads=3,
        num_key_value_heads=None,
        head_dim=None,
        num_experts_per_tok=None,
        use_cache=False,
        custom_field=7,
    )
    assert config.head_dim == 32
    assert config.num_key_value_heads == 3
    assert config.num_experts_per_tok is None
    assert config.use_cache is False
    assert config.custom_field == 7
