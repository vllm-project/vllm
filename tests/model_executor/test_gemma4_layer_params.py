# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.gemma4 import get_layer_params


def test_get_layer_params_returns_per_layer_attention_and_ffn_params() -> None:
    config = SimpleNamespace(
        num_hidden_layers=4,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        head_dim=128,
        global_head_dim=256,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        attention_k_eq_v=True,
        intermediate_size=4096,
        num_kv_shared_layers=2,
        use_double_wide_mlp=True,
    )

    params = get_layer_params(config)

    assert [p["head_dim"] for p in params] == [128, 256, 128, 256]
    assert [p["num_kv_heads"] for p in params] == [8, 2, 8, 2]
    assert [p["has_v_proj"] for p in params] == [True, False, True, False]
    assert [p["intermediate_size"] for p in params] == [4096, 4096, 8192, 8192]
    assert [p["is_kv_shared"] for p in params] == [False, False, True, True]
    assert [p["kv_shared_target"] for p in params] == [None, None, 0, 1]
    assert [p["is_sliding"] for p in params] == [True, False, True, False]


def test_get_layer_params_supports_defaults_and_nested_config() -> None:
    text_config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        head_dim=64,
        num_key_value_heads=4,
        intermediate_size=1024,
    )
    config = SimpleNamespace(text_config=text_config)

    params = get_layer_params(config)
    expected = {
        "head_dim": 64,
        "num_kv_heads": 4,
        "has_v_proj": True,
        "intermediate_size": 1024,
        "is_kv_shared": False,
        "kv_shared_target": None,
        "is_sliding": True,
    }
    assert params == [expected, expected | {"is_sliding": False}]


def test_get_layer_params_rejects_shared_layer_without_matching_target() -> None:
    config = SimpleNamespace(
        num_hidden_layers=3,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        head_dim=64,
        num_key_value_heads=4,
        intermediate_size=1024,
        num_kv_shared_layers=1,
    )

    with pytest.raises(
        ValueError,
        match="layer 2 has no non-shared full_attention layer",
    ):
        get_layer_params(config)
