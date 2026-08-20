# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs import MiniMaxM3TextConfig


def test_load_local_minimax_m3_text_config(tmp_path: Path):
    config = MiniMaxM3TextConfig(
        num_hidden_layers=4,
        moe_layer_freq=[0, 0, 0, 1],
        sparse_attention_config={},
    )
    config.save_pretrained(tmp_path)

    loaded = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(loaded, MiniMaxM3TextConfig)
    assert loaded.num_hidden_layers == 4
    assert loaded.moe_layer_freq == [0, 0, 0, 1]
