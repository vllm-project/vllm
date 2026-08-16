# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig


def _write_config(path, config):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))


@pytest.fixture
def lfm2_dspark_configs(tmp_path):
    target_path = tmp_path / "target"
    draft_path = tmp_path / "draft"
    _write_config(
        target_path,
        {
            "architectures": ["Lfm2ForCausalLM"],
            "model_type": "lfm2",
            "hidden_size": 2048,
            "intermediate_size": 10752,
            "num_hidden_layers": 30,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128000,
            "max_position_embeddings": 131072,
            "conv_L_cache": 3,
            "conv_dim": 2048,
            "layer_types": ["conv"] * 20 + ["full_attention"] * 10,
        },
    )
    _write_config(
        draft_path,
        {
            "architectures": ["Lfm2DSparkDraftModel"],
            "model_type": "qwen3",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 128000,
            "max_position_embeddings": 128000,
            "block_size": 9,
            "dflash_config": {
                "target_layer_ids": [2, 9, 17, 21, 27],
                "num_target_layers": 30,
            },
            "markov_rank": 256,
            "rope_is_neox_style": False,
            "enable_confidence_head": True,
        },
    )
    return target_path, draft_path


def _make_speculative_config(target_path, draft_path, **overrides):
    target_config = ModelConfig(
        model=str(target_path), tokenizer_mode="skip", max_model_len=128000
    )
    return SpeculativeConfig(
        model=str(draft_path),
        target_model_config=target_config,
        target_parallel_config=ParallelConfig(),
        **overrides,
    )


@pytest.mark.cpu_test
def test_lfm2_dspark_checkpoint_config_is_normalized(lfm2_dspark_configs):
    target_path, draft_path = lfm2_dspark_configs

    config = _make_speculative_config(target_path, draft_path, dspark_draft_topk=1024)

    assert config.method == "dspark"
    assert config.parallel_drafting
    assert config.num_speculative_tokens == 9
    assert config.draft_model_config.architectures == ["Lfm2DSparkDraftModel"]
    draft_hf_config = config.draft_model_config.hf_config
    assert draft_hf_config.n_predict == 9
    assert not draft_hf_config.is_neox_style
    assert draft_hf_config.confidence_head_with_markov
    assert draft_hf_config.dspark_draft_topk == 1024


@pytest.mark.cpu_test
def test_lfm2_dspark_rejects_topk_above_vocabulary(lfm2_dspark_configs):
    target_path, draft_path = lfm2_dspark_configs

    with pytest.raises(ValueError, match="draft vocabulary size"):
        _make_speculative_config(target_path, draft_path, dspark_draft_topk=128001)
