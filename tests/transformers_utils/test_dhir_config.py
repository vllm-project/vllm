# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from huggingface_hub.errors import (
    StrictDataclassClassValidationError,
    StrictDataclassFieldValidationError,
)

from vllm.config import ModelConfig
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.dhir import DhirConfig, DhirMoeConfig

# Released checkpoints ship remote code and a top-level `rope_theta`.
HUB_CONFIGS = {
    "dhir": {
        "architectures": ["DhirForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_dhir.DhirConfig"},
        "rope_theta": 1000000.0,
    },
    "dhir_moe": {
        "architectures": ["DhirMoeForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_dhir_moe.DhirMoeConfig"},
        "rope_theta": 10000000.0,
        "num_experts": 12,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
    },
}


@pytest.mark.parametrize(
    "model_type,config_cls", [("dhir", DhirConfig), ("dhir_moe", DhirMoeConfig)]
)
def test_dhir_hub_config_loads_without_remote_code(tmp_path, model_type, config_cls):
    hub_config = {"model_type": model_type, **HUB_CONFIGS[model_type]}
    (tmp_path / "config.json").write_text(json.dumps(hub_config))

    config = get_config(tmp_path, trust_remote_code=False)

    assert type(config) is config_cls
    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": hub_config["rope_theta"],
    }
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, max_model_len=128
    )
    assert model_config.architectures == hub_config["architectures"]
    assert model_config.get_head_size() == 128


@pytest.mark.parametrize(
    "config_cls,overrides,match",
    [
        (DhirConfig, {"num_key_value_heads": 3}, "divisible"),
        (DhirConfig, {"hidden_act": "gelu"}, "silu"),
        (DhirMoeConfig, {"num_experts_per_tok": 13}, "num_experts_per_tok"),
    ],
)
def test_dhir_rejects_unsupported_config(config_cls, overrides, match):
    with pytest.raises(
        (
            ValueError,
            StrictDataclassClassValidationError,
            StrictDataclassFieldValidationError,
        ),
        match=match,
    ):
        config_cls(**overrides)
