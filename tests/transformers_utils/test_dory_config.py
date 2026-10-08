# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.dory import DoryConfig


def test_dory_loads_without_remote_code_and_preserves_head_size(tmp_path):
    config = DoryConfig(
        n_input_layers=16,
        n_recurrent_layers=40,
        n_output_layers=16,
        architectures=["DoryForCausalLM"],
    )
    config.save_pretrained(tmp_path)
    restored = get_config(tmp_path, trust_remote_code=False)
    assert isinstance(restored, DoryConfig)
    assert restored.recurrent_kv_cache_mode == "per_loop"
    assert DoryConfig.from_dict(restored.to_dict()).to_dict() == restored.to_dict()
    # 2560 / 16 = 160 is NOT the checkpoint's attention head dimension.
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, max_model_len=128
    )
    assert model_config.get_head_size() == 256


def test_dory_last_loop_cache_can_be_selected_with_hf_overrides(tmp_path):
    DoryConfig(architectures=["DoryForCausalLM"]).save_pretrained(tmp_path)
    config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=128,
        hf_overrides={"recurrent_kv_cache_mode": "last_loop"},
    ).hf_config
    config.validate_architecture()
    assert config.recurrent_kv_cache_mode == "last_loop"
    assert DoryConfig.from_dict(config.to_dict()).recurrent_kv_cache_mode == "last_loop"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"n_recurrent_layers": 40}, "sum to num_hidden_layers"),
        ({"n_recurrent_loops": 0}, "n_recurrent_loops"),
        ({"n_recurrent_loops": 1.5}, "n_recurrent_loops"),
        ({"recurrent_kv_cache_mode": "loop1"}, "recurrent_kv_cache_mode"),
        ({"hybrid_layer_pattern": "M" * 72}, "hybrid_layer_pattern"),
        ({"rope_profile_layers": [1]}, "rope_profile_layers"),
        ({"rope_profile_layers": [3] * 72}, "rope_profile_layers"),
        ({"no_rope_layers": []}, "no_rope_layers"),
        ({"rope_profile_layers": [0] * 72}, "does not support NoPE"),
        ({"no_rope_layers": [1] * 72}, "does not support NoPE"),
        ({"head_dim": 160}, "must agree"),
        ({"swa_head_dim": 128}, "matching"),
        ({"swa_window_size": None}, "swa_window_size"),
        ({"partial_rotary_factor": 0}, "Rotary dimensions"),
        ({"output_transform": "adaptive_slerp"}, "replace output transform"),
        ({"ngpt": False}, "ngpt=True"),
        ({"attention_bias": True}, "bias-free"),
        ({"initial_state": "zero"}, "input_copy"),
        ({"output_transform": "gated"}, "output transform"),
    ],
)
def test_dory_rejects_incompatible_architecture(overrides, match):
    with pytest.raises(ValueError, match=match):
        DoryConfig(**overrides)
