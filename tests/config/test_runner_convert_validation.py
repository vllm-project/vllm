# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from vllm.config import ModelConfig


def test_classify_rejects_generate_runner(tmp_path: Path):
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "vocab_size": 100,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="convert classify.*runner pooling"):
        ModelConfig(
            model=str(tmp_path),
            runner="generate",
            convert="classify",
            skip_tokenizer_init=True,
        )
