# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.k3_dspark import K3DSparkConfig


def _write_dspark_config(path, **overrides):
    path.mkdir()
    config = {
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "num_hidden_layers": 5,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 163840,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 32768,
        "rope_theta": 50000.0,
        "num_target_layers": 5,
        "target_hidden_size": 7168,
        "target_num_hidden_layers": 93,
        "target_layer_ids": [2, 23, 47, 71, 89],
        "markov_rank": 256,
        "draft_vocab_size": 163840,
        "torch_dtype": "bfloat16",
    }
    config.update(overrides)
    (path / "config.json").write_text(json.dumps(config))


def _write_target_config(path):
    path.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "num_hidden_layers": 93,
        "num_attention_heads": 56,
        "num_key_value_heads": 8,
        "vocab_size": 163840,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
    }
    (path / "config.json").write_text(json.dumps(config))


def test_dspark_mla_config_loads_from_local_json(tmp_path):
    draft_path = tmp_path / "draft"
    _write_dspark_config(draft_path)

    config = get_config(draft_path, trust_remote_code=False)

    assert isinstance(config, K3DSparkConfig)
    assert config.model_type == "k3_dspark"
    assert config.architectures == ["K3DSparkModel"]
    assert config.hidden_act == "silu"
    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 50000.0,
    }
    assert config.n_routed_experts == 0
    assert config.draft_vocab_size == config.vocab_size


@pytest.mark.parametrize(
    "overrides",
    [
        {"mla_use_nope": True},
        {"mla_use_output_gate": True},
        {"mla_use_qk_norm": True},
        {"dspark_bonus_anchor": True},
        {"q_lora_rank": None},
        {"draft_vocab_size": 8192},
        {"target_layer_ids": []},
        {"num_target_layers": 4},
    ],
)
def test_dspark_mla_rejects_unsupported_checkpoint_options(tmp_path, overrides):
    draft_path = tmp_path / "draft"
    _write_dspark_config(draft_path, **overrides)

    with pytest.raises(ValueError, match="MLA DSpark"):
        get_config(draft_path, trust_remote_code=False)


@pytest.mark.parametrize(
    ("num_attention_heads", "expected_local_heads"),
    [(64, 8), (96, 12)],
    ids=["released-64-head", "block5-96-head"],
)
def test_dspark_mla_uses_latent_kv_geometry(
    tmp_path, num_attention_heads, expected_local_heads
):
    draft_path = tmp_path / "draft"
    _write_dspark_config(
        draft_path,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
    )
    model_config = ModelConfig(
        model=str(draft_path),
        tokenizer_mode="skip",
        runner="draft",
        max_model_len=32768,
    )

    assert model_config.is_deepseek_mla
    assert model_config.use_mla
    assert model_config.get_head_size() == 576
    # external_launcher skips ParallelConfig's local-GPU-count check so the
    # config logic can be exercised at TP8 on a single-GPU test node.
    parallel_config = ParallelConfig(
        tensor_parallel_size=8, distributed_executor_backend="external_launcher"
    )
    assert model_config.get_num_kv_heads(parallel_config) == 1
    assert model_config.get_num_attention_heads(parallel_config) == expected_local_heads
    assert model_config.get_num_experts() == 0


def test_dspark_mla_speculative_config_preserves_architecture(tmp_path):
    target_path = tmp_path / "target"
    draft_path = tmp_path / "draft"
    _write_target_config(target_path)
    _write_dspark_config(draft_path)
    target_config = ModelConfig(
        model=str(target_path), tokenizer_mode="skip", max_model_len=32768
    )
    speculative_config = SpeculativeConfig(
        model=str(draft_path),
        method="dspark",
        num_speculative_tokens=8,
        target_model_config=target_config,
        target_parallel_config=ParallelConfig(),
    )

    assert speculative_config.parallel_drafting
    assert speculative_config.draft_model_config.architectures == ["K3DSparkModel"]
    assert speculative_config.draft_model_config.hf_config.model_type == "k3_dspark"
    assert speculative_config.draft_model_config.use_mla
