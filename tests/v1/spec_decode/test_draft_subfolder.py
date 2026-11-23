# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Without `model`, the draft is taken from `subfolder` of the target
checkpoint, which defaults to the method name, and a missing draft fails
clearly."""

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


def _make_spec(
    tmp_path, draft_dir: str | None, subfolder: str | None
) -> SpeculativeConfig:
    (tmp_path / "config.json").write_text(json.dumps(_TARGET_CONFIG))
    if draft_dir is not None:
        (tmp_path / draft_dir).mkdir()
        (tmp_path / draft_dir / "config.json").write_text(json.dumps(_DRAFT_CONFIG))
    target = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=128,
    )
    return SpeculativeConfig(
        method="eagle3",
        subfolder=subfolder,
        num_speculative_tokens=3,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "draft_dir,subfolder",
    [("eagle3", None), ("draft", "draft")],
    ids=["method-default", "explicit"],
)
def test_draft_from_target_subfolder(tmp_path, draft_dir, subfolder):
    spec = _make_spec(tmp_path, draft_dir, subfolder)

    assert spec.model == str(tmp_path / draft_dir)
    assert spec.draft_model_config.hf_config.architectures == ["Eagle3LlamaForCausalLM"]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "subfolder,match",
    [
        ("draft", "no draft checkpoint in the 'draft/'"),
        (None, "without speculative model"),
    ],
    ids=["explicit", "method-default"],
)
def test_missing_draft_subfolder_raises(tmp_path, subfolder, match):
    with pytest.raises(ValueError, match=match):
        _make_spec(tmp_path, None, subfolder)


@pytest.mark.cpu_test
def test_subfolder_with_model_raises():
    """`subfolder` is relative to the target, so a resolved config rebuilt from
    its fields fails instead of looking in a nested subfolder."""
    with pytest.raises(ValueError, match="cannot be combined with model"):
        SpeculativeConfig(method="eagle3", model="draft", subfolder="eagle3")
