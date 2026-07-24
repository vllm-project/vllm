# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.orthrus import OrthrusConfig

pytestmark = pytest.mark.skip_global_cleanup


def test_orthrus_config_registered_without_remote_code(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["OrthrusLM"],
                "model_type": "orthrus",
                "vocab_size": 151936,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "max_position_embeddings": 40960,
                "block_size": 16,
                "mask_token_id": 151665,
            }
        )
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, OrthrusConfig)
    assert config.model_type == "orthrus"
    assert config.block_size == 16
    assert config.mask_token_id == 151665
