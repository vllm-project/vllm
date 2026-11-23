# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`subfolder` without `model` takes the draft that the target checkpoint ships
in that subfolder, and fails clearly when it is missing."""

import json

import pytest

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig

_TARGET_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
}

_DRAFT_CONFIG = {
    **_TARGET_CONFIG,
    "architectures": ["Eagle3LlamaForCausalLM"],
    "num_hidden_layers": 1,
    "draft_vocab_size": 500,
    "target_hidden_size": 64,
    "eagle_config": {
        "use_aux_hidden_state": True,
        "eagle_aux_hidden_state_layer_ids": [1, 2, 3],
    },
}


def _make_spec(tmp_path, with_draft: bool) -> SpeculativeConfig:
    (tmp_path / "config.json").write_text(json.dumps(_TARGET_CONFIG))
    if with_draft:
        (tmp_path / "eagle3").mkdir()
        (tmp_path / "eagle3" / "config.json").write_text(json.dumps(_DRAFT_CONFIG))
    target = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=128,
    )
    return SpeculativeConfig(
        method="eagle3",
        subfolder="eagle3",
        num_speculative_tokens=3,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )


@pytest.mark.cpu_test
def test_draft_from_target_subfolder(tmp_path):
    spec = _make_spec(tmp_path, with_draft=True)

    assert spec.model == str(tmp_path / "eagle3")
    assert spec.draft_model_config.hf_config.architectures == ["Eagle3LlamaForCausalLM"]


@pytest.mark.cpu_test
def test_missing_draft_subfolder_raises(tmp_path):
    with pytest.raises(ValueError, match="no draft checkpoint in the 'eagle3/'"):
        _make_spec(tmp_path, with_draft=False)
