# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test config resolution for checkpoints whose `config.json` omits
`model_type` while still declaring `auto_map`."""

import json
import tempfile
from pathlib import Path

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.maple import MapleConfig

# Trimmed from `deepgrove/maple-preview`, which serializes `auto_map` but never
# `model_type`, so `AutoConfig` has nothing to dispatch on.
_CONFIG_WITHOUT_MODEL_TYPE = {
    "architectures": ["MapleForCausalLM"],
    "auto_map": {
        "AutoConfig": "configuration_maple.MapleConfig",
        "AutoModelForCausalLM": "modeling_maple.MapleForCausalLM",
    },
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
}


@pytest.mark.cpu_test
def test_model_type_recovered_from_architectures():
    """`architectures` must be enough to reach the registered config class, so
    a checkpoint that only declares `model_type` on its remote config class
    does not require `trust_remote_code`."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(_CONFIG_WITHOUT_MODEL_TYPE))

        config = get_config(tmp_dir, trust_remote_code=False)

    assert isinstance(config, MapleConfig)
    assert config.model_type == "maple"
    assert config.num_hidden_layers == 4
    # Defaults only the vLLM config class knows about must still be filled in.
    assert config.swiglu_limit == 7.0
    assert len(config.layer_types) == 4


@pytest.mark.cpu_test
def test_interleaved_maple_does_not_set_a_global_sliding_window():
    """Full-attention layers must not inherit the local window from the cache.

    `Attention` falls back to `cache_config.sliding_window` when a layer has no
    per-layer value. Setting it for Maple would silently turn every fourth,
    full-attention layer into a sliding-window layer.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(_CONFIG_WITHOUT_MODEL_TYPE))

        vllm_config = EngineArgs(
            model=tmp_dir,
            tokenizer=tmp_dir,
            max_model_len=64,
        ).create_engine_config()

    assert vllm_config.model_config.hf_config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert vllm_config.cache_config.sliding_window is None
