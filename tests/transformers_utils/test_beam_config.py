# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test Beam config aliases, hybrid attention, and serialization."""

from typing import Any

import pytest

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.beam import (
    BeamConfig,
    _canonical_dtype,
)

# Synthetic configuration with hybrid attention and grouped routing.
_NUM_LAYERS = 4
_GLOBAL_ATTENTION_LAYER_IDS = frozenset({1, 3})

_SYNTHETIC_CONFIG: dict[str, Any] = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": _NUM_LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "hidden_act": "silu",
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attention_output_gate": False,
    "sliding_window": 32,
    "layer_types": [
        "full_attention"
        if layer_id in _GLOBAL_ATTENTION_LAYER_IDS
        else "sliding_attention"
        for layer_id in range(_NUM_LAYERS)
    ],
    "moe_intermediate_size": 128,
    "num_experts_per_tok": 2,
    "num_experts": 8,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "n_group": 2,
    "topk_group": 1,
    "routed_scaling_factor": 1.5,
    "topk_method": "noaux_tc",
    "mlp_only_layers": [0],
    "rope_theta": 10000,
    "rope_scaling": None,
    "nope_layer_ids": sorted(_GLOBAL_ATTENTION_LAYER_IDS),
    "mlp_depth_scaled_residual": True,
    "self_attn_depth_scaled_residual": True,
    "final_logit_softcapping": 30.0,
    "moe_router_dtype": "fp32",
    "ssmax": True,
    "ssmax_offset": 1,
    "ssmax_divide_by_log_offset": False,
    "ssmax_scale_mode": "root10",
    "use_paramfree_qk_norm": True,
    "enable_sandwich_norm": True,
}


def test_synthetic_config_resolves():
    config = BeamConfig(**_SYNTHETIC_CONFIG)
    assert config.model_type == "beam"
    assert config.architectures == ["BeamForCausalLM"]
    assert config.num_hidden_layers == 4
    assert config.hidden_size == 64
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 2
    assert config.head_dim == 16
    assert config.num_experts == 8
    assert config.moe_intermediate_size == 128
    assert config.nope_layer_ids == [1, 3]
    assert config.mlp_only_layers == [0]
    assert config.final_logit_softcapping == 30.0
    assert config.moe_router_dtype == "float32"
    assert config.ssmax and config.ssmax_scale_mode == "root10"
    assert config.use_paramfree_qk_norm
    assert config.enable_sandwich_norm
    for layer_id, layer_type in enumerate(config.layer_types):
        expected = (
            "full_attention"
            if layer_id in _GLOBAL_ATTENTION_LAYER_IDS
            else "sliding_attention"
        )
        assert layer_type == expected


def test_config_aliases_resolve():
    """Config aliases resolve to the fields used by the model."""
    no_rope = [1] * 4
    for i in (1, 3):
        no_rope[i] = 0
    config = BeamConfig(
        num_hidden_layers=4,
        no_rope_layers=no_rope,
        num_experts=8,
        router_dtype="fp32",
        rope_parameters={"rope_theta": 5000.0, "rope_type": "default"},
    )
    assert config.nope_layer_ids == [1, 3]
    assert config.num_experts == 8
    assert config.moe_router_dtype == "float32"
    assert config.rope_theta == 5000.0


def test_unsupported_dtype_keys_are_dropped():
    """Unsupported dtype options are excluded from the serialized config."""
    config = BeamConfig(
        num_hidden_layers=4, residual_in_fp32=True, lm_head_dtype="fp32"
    )
    for key in ("residual_in_fp32", "lm_head_dtype", "fp32_residual_connection"):
        assert not hasattr(config, key)
        assert key not in config.to_dict()


def test_layer_types_derived_when_omitted():
    """A missing `layer_types` must not leave the SWA/full split implicit.

    vLLM only skips populating the global `CacheConfig.sliding_window` when
    `layer_types` says the model is partly sliding; without it the
    full-attention layers pick that global value up as a fallback and silently
    get a window.
    """
    no_rope = [1] * 8
    for i in (3, 7):
        no_rope[i] = 0
    config = BeamConfig(num_hidden_layers=8, no_rope_layers=no_rope, sliding_window=512)
    assert config.layer_types == [
        "full_attention" if i in (3, 7) else "sliding_attention" for i in range(8)
    ]


def test_explicit_args_beat_aliases():
    config = BeamConfig(
        num_hidden_layers=4,
        nope_layer_ids=[1],
        no_rope_layers=[1, 1, 0, 1],
        moe_router_dtype="bf16",
        router_dtype="float32",
    )
    assert config.nope_layer_ids == [1]
    assert config.moe_router_dtype == "bfloat16"


def test_to_dict_roundtrip_keeps_canonical_fields():
    config = BeamConfig(**_SYNTHETIC_CONFIG)
    serialized = config.to_dict()
    assert serialized["num_experts"] == 8
    assert "num_local_experts" not in serialized
    rebuilt = BeamConfig(**serialized)
    assert rebuilt.num_experts == 8
    assert rebuilt.nope_layer_ids == config.nope_layer_ids
    assert rebuilt.moe_router_dtype == config.moe_router_dtype


def test_noncanonical_expert_count_is_rejected():
    with pytest.raises(ValueError, match="requires num_experts"):
        BeamConfig(num_local_experts=8)


def test_nested_rope_checkpoint_loads_and_infers_nope(tmp_path):
    """Release metadata must survive generic loading and config round trips."""
    import json

    checkpoint = {
        "model_type": "beam",
        "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 5000.0},
            "full_attention": None,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert config.nope_layer_ids == [1]
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 5000.0}
    rebuilt = BeamConfig(**config.to_dict())
    assert rebuilt.nope_layer_ids == [1]
    assert rebuilt.rope_parameters == config.rope_parameters


def test_release_config_omitted_features_match_reference(tmp_path):
    """A checkpoint config that omits the fixed-behavior switches resolves to them."""
    import json

    num_layers, nope_layers = 12, (3, 7, 11)
    checkpoint = {
        "model_type": "beam",
        "architectures": ["BeamForCausalLM"],
        "num_hidden_layers": num_layers,
        "num_experts": 32,
        "n_group": 4,
        "topk_group": 2,
        "routed_scaling_factor": 2.5,
        "sliding_window": 128,
        "layer_types": [
            "full_attention" if i in nope_layers else "sliding_attention"
            for i in range(num_layers)
        ],
        "mlp_layer_types": ["dense"] * 2 + ["sparse"] * (num_layers - 2),
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint))
    config = get_config(str(tmp_path), trust_remote_code=False)
    for resolved in (config, BeamConfig(**config.to_dict())):
        assert resolved.attention_output_gate
        assert resolved.attention_output_gate_bias
        assert resolved.enable_sandwich_norm
        assert resolved.self_attn_depth_scaled_residual
        assert resolved.mlp_depth_scaled_residual
        assert resolved.ssmax
        assert resolved.ssmax_scale_mode == "root10"
        assert resolved.scoring_func == "sigmoid"
        assert resolved.topk_method == "noaux_tc"
        assert resolved.nope_layer_ids == list(nope_layers)
        assert resolved.mlp_only_layers == [0, 1]


@pytest.mark.parametrize(
    "explicit", [{"nope_layer_ids": []}, {"no_rope_layers": [1, 1]}]
)
def test_explicit_rope_schedule_beats_layer_types(explicit):
    config = BeamConfig(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        **explicit,
    )
    assert config.nope_layer_ids == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"layer_types": ["full_attention"]},
        {"layer_types": ["full_attention", "invalid"]},
        {"no_rope_layers": [1]},
        {"no_rope_layers": [1, 2]},
        {"nope_layer_ids": [2]},
        {"nope_layer_ids": [1, 1]},
    ],
)
def test_invalid_attention_schedule_is_rejected(overrides):
    with pytest.raises(ValueError):
        BeamConfig(num_hidden_layers=2, **overrides)


def test_dtype_spellings_are_canonicalized():
    """Short spellings must resolve to the keys of `STR_DTYPE_TO_TORCH_DTYPE`."""
    assert _canonical_dtype(None) is None
    assert _canonical_dtype("fp32") == "float32"
    assert _canonical_dtype("float32") == "float32"
    assert _canonical_dtype("bf16") == "bfloat16"
    assert _canonical_dtype("bfloat16") == "bfloat16"
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _canonical_dtype("fp16")


def test_mlp_layer_types_derives_dense_layers():
    """Newer checkpoints list per-layer feed-forward types instead of
    `mlp_only_layers`; the model keeps consuming the dense-layer list."""
    config = BeamConfig(
        num_hidden_layers=4, mlp_layer_types=["dense", "sparse", "sparse", "dense"]
    )
    assert config.mlp_only_layers == [0, 3]
    assert config.mlp_layer_types == ["dense", "sparse", "sparse", "dense"]


def test_mlp_only_layers_derives_layer_types():
    config = BeamConfig(num_hidden_layers=4, mlp_only_layers=[0])
    assert config.mlp_layer_types == ["dense", "sparse", "sparse", "sparse"]


def test_agreeing_mlp_schemas_are_accepted_and_round_trip():
    config = BeamConfig(
        num_hidden_layers=3,
        mlp_only_layers=[0, 1],
        mlp_layer_types=["dense", "dense", "sparse"],
    )
    rebuilt = BeamConfig(**config.to_dict())
    assert rebuilt.mlp_only_layers == [0, 1]
    assert rebuilt.mlp_layer_types == ["dense", "dense", "sparse"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"mlp_only_layers": [0], "mlp_layer_types": ["sparse", "dense", "sparse"]},
        {"mlp_layer_types": ["dense", "sparse"]},
        {"mlp_layer_types": ["dense", "moe", "sparse"]},
        {"mlp_only_layers": [0, 0]},
        {"mlp_only_layers": [3]},
    ],
)
def test_invalid_mlp_schedules_are_rejected(overrides):
    with pytest.raises(ValueError):
        BeamConfig(num_hidden_layers=3, **overrides)
