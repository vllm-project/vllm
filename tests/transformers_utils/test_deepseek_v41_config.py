# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.model import ModelConfig
from vllm.transformers_utils.config import get_hf_text_config
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config


@pytest.mark.cpu_test
def test_hf_overrides_text_config_merges_into_flattened_fields():
    config = DeepseekV41Config(
        text_config={"num_attention_heads": 128, "num_experts_per_tok": 6}
    )

    ModelConfig._apply_dict_overrides(
        ModelConfig.__new__(ModelConfig),
        config,
        {"text_config": {"num_experts_per_tok": 8}},
    )

    assert config.num_experts_per_tok == 8
    assert config.num_attention_heads == 128
    assert get_hf_text_config(config) is config
