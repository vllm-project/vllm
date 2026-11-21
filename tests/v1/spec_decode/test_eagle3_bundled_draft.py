# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`method: eagle3` without `model` takes the draft that the target checkpoint
ships in its `eagle3/` subfolder, and still fails clearly without one."""

import json

import pytest

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.eagle.config import EAGLE3_SUBFOLDER

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


def _write_target(tmp_path, with_draft: bool):
    (tmp_path / "config.json").write_text(json.dumps(_TARGET_CONFIG))
    if with_draft:
        draft_dir = tmp_path / EAGLE3_SUBFOLDER
        draft_dir.mkdir()
        (draft_dir / "config.json").write_text(json.dumps(_DRAFT_CONFIG))
    return ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=128,
    )


@pytest.mark.cpu_test
def test_bundled_eagle3_draft_is_the_default_model(tmp_path):
    target = _write_target(tmp_path, with_draft=True)

    spec = SpeculativeConfig(
        method="eagle3",
        num_speculative_tokens=3,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )

    assert spec.model == str(tmp_path / EAGLE3_SUBFOLDER)
    assert spec.method == "eagle3"
    assert spec.draft_model_config.hf_config.architectures == ["Eagle3LlamaForCausalLM"]


@pytest.mark.cpu_test
def test_eagle3_without_model_or_bundled_draft_raises(tmp_path):
    target = _write_target(tmp_path, with_draft=False)

    with pytest.raises(ValueError, match="without speculative model"):
        SpeculativeConfig(
            method="eagle3",
            num_speculative_tokens=3,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )
