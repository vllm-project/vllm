# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3MTPConfig


def test_standalone_mtp_config_keeps_legacy_per_layer_fields(tmp_path: Path):
    """Standalone MTP checkpoints copy the main model's 60-layer lists."""
    config = {
        "architectures": ["MiniMaxM3MTP"],
        "model_type": "minimax_m3_mtp",
        "num_hidden_layers": 1,
        "moe_layer_freq": [0] * 3 + [1] * 57,
        "sparse_attention_config": {
            "sparse_attention_freq": [0] * 3 + [1] * 57,
            "sparse_num_index_heads": 4,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, MiniMaxM3MTPConfig)
    assert config.layer_types == ["full_attention"]
    assert config.mlp_layer_types == ["dense"]
    assert config.index_n_heads == 4
