# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
from huggingface_hub.errors import (
    StrictDataclassClassValidationError,
    StrictDataclassFieldValidationError,
)

from vllm.config import ModelConfig, ParallelConfig
from vllm.exceptions import VLLMValidationError
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.dory1_5 import (
    Dory1_5Config,
)


@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
def test_dory1_5_native_config_roundtrip_and_hybrid_layout(tmp_path, cache_mode):
    original = Dory1_5Config(
        architectures=["Dory1_5ForCausalLM"],
        recurrent_kv_cache_mode=cache_mode,
    )
    original.save_pretrained(tmp_path)
    restored = get_config(tmp_path, trust_remote_code=False)
    assert isinstance(restored, Dory1_5Config)
    assert Dory1_5Config.from_dict(restored.to_dict()).to_dict() == restored.to_dict()
    assert restored.layer_types == original.layer_types
    model = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=128,
        hf_overrides={"n_recurrent_loops": 2},
    )
    assert model.hf_config.n_recurrent_loops == 2
    assert model.hf_config.recurrent_kv_cache_mode == cache_mode
    assert model.get_head_size() == 128
    assert model.is_hybrid
    parallel = ParallelConfig()
    assert model.get_num_layers_by_block_type(parallel, "attention") == 6
    assert model.get_num_layers_by_block_type(parallel, "linear_attention") == 23


def test_dory1_5_accepts_hf_special_token_types():
    config = Dory1_5Config(eos_token_id=[11, 12], bos_token_id=None, pad_token_id=None)
    assert config.eos_token_id == [11, 12]
    assert config.bos_token_id is None
    assert config.pad_token_id is None


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"n_recurrent_loops": 0}, "must be positive"),
        ({"n_recurrent_loops": 1.5}, "n_recurrent_loops"),
        ({"recurrent_kv_cache_mode": "last"}, "recurrent_kv_cache_mode"),
        ({"n_recurrent_layers": 22}, "partitions"),
        ({"layer_types": []}, "num_hidden_layers"),
        ({"layer_types": ["moe"] * 52}, "Dory1.5 layers"),
        ({"linear_num_key_heads": 7}, "divisible"),
        ({"head_dim": 0}, "dimensions"),
        ({"loop_conditioning_hidden_size": 255}, "must be even"),
        ({"hidden_act": "silu"}, "relu2"),
    ],
)
def test_dory1_5_rejects_incompatible_architecture(overrides, match):
    with pytest.raises(
        (
            ValueError,
            StrictDataclassClassValidationError,
            StrictDataclassFieldValidationError,
        ),
        match=match,
    ):
        Dory1_5Config(**overrides)


@pytest.mark.parametrize("count", [True, False, 0, 5, "2", 2.0, None])
def test_request_loop_budget_rejects_invalid_counts(count):
    config = Dory1_5Config(n_recurrent_loops=4)
    with pytest.raises(VLLMValidationError, match="num_loops"):
        config.request_cache_salt({"num_loops": count}, "caller")


def test_request_cache_salt_preserves_default_and_isolates_budgets():
    config = Dory1_5Config(n_recurrent_loops=4)
    assert config.request_cache_salt(None, "caller") == "caller"
    assert config.request_cache_salt({"num_loops": 4}, "caller") == "caller"
    salts = {
        config.request_cache_salt({"num_loops": count}, caller)
        for count in range(1, 5)
        for caller in (None, "caller", "other")
    }
    assert len(salts) == 12


@pytest.mark.parametrize(
    "marker",
    [
        {},
        {"version": 2, "allocation": "x.json"},
        {"version": True, "allocation": "x.json"},
        {"version": 1},
        {"version": 1, "allocation": ""},
        {"version": 1, "allocation": 1},
    ],
)
def test_invalid_per_tensor_quantization(marker):
    with pytest.raises(
        StrictDataclassClassValidationError, match="per_tensor_quantization"
    ):
        Dory1_5Config(per_tensor_quantization=marker).validate()


def test_per_tensor_quantization_roundtrip(tmp_path):
    marker = {"version": 1, "allocation": "quantization_allocation.json"}
    config = Dory1_5Config(per_tensor_quantization=marker)
    config.save_pretrained(tmp_path)
    assert Dory1_5Config.from_pretrained(tmp_path).per_tensor_quantization == marker
