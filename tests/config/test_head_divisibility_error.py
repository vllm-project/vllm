# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the attention-head / tensor-parallel divisibility error message."""

import json

import pytest

from vllm.config import ModelConfig, ParallelConfig


def _make_model_config(tmp_path, num_attention_heads: int = 64) -> ModelConfig:
    """Build a ModelConfig from a minimal local Llama config (no download)."""
    config_dict = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 2,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "vocab_size": 32000,
        "max_position_embeddings": 8192,
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict))
    return ModelConfig(model=str(tmp_path), runner="generate", max_model_len=1024)


def test_head_divisibility_error_is_actionable(tmp_path):
    """A TP size that does not divide the attention heads should raise an
    actionable error: list the valid `--tensor-parallel-size` values and
    suggest pipeline parallelism (which shards by layers) when supported."""
    model_config = _make_model_config(tmp_path, num_attention_heads=64)

    with pytest.raises(ValueError) as exc_info:
        model_config.verify_with_parallel_config(ParallelConfig(tensor_parallel_size=3))

    message = str(exc_info.value)
    assert "Total number of attention heads (64)" in message
    assert "must be divisible by tensor parallel size (3)" in message
    assert (
        "`--tensor-parallel-size` values are valid: [1, 2, 4, 8, 16, 32, 64]" in message
    )
    # Llama implements SupportsPP, so the suggestion should be offered.
    assert "consider `--pipeline-parallel-size` instead" in message


def test_head_divisibility_error_valid_tp_size_passes(tmp_path):
    """A divisor TP size should not raise."""
    model_config = _make_model_config(tmp_path, num_attention_heads=64)
    # 64 % 4 == 0 -> no error.
    model_config.verify_with_parallel_config(ParallelConfig(tensor_parallel_size=4))
