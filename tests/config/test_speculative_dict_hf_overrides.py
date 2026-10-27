# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig


def test_compose_draft_hf_overrides_with_dict():
    """Verify compose_draft_hf_overrides properly merges nested dict overrides."""
    hf_config = PretrainedConfig()
    hf_config.architectures = ["MockForCausalLM"]
    hf_config.max_position_embeddings = 262144
    hf_config.text_config = {
        "max_position_embeddings": 262144,
        "rope_parameters": {"factor": 1.0, "rope_type": "default"},
    }

    dict_overrides = {
        "text_config": {
            "max_position_embeddings": 409600,
            "rope_parameters": {"factor": 1.5625, "rope_type": "yarn"},
        }
    }

    composed_fn = SpeculativeConfig.compose_draft_hf_overrides(
        dict_overrides, apply_dict_overrides=True
    )
    assert callable(composed_fn)

    updated_config = composed_fn(hf_config)
    assert updated_config.text_config["max_position_embeddings"] == 409600
    assert updated_config.text_config["rope_parameters"]["factor"] == 1.5625
    assert updated_config.text_config["rope_parameters"]["rope_type"] == "yarn"


def test_compose_draft_hf_overrides_with_callable():
    """Verify compose_draft_hf_overrides preserves existing callable behavior."""
    hf_config = PretrainedConfig()
    hf_config.architectures = ["MockForCausalLM"]
    hf_config.max_position_embeddings = 2048

    def custom_fn(config):
        config.max_position_embeddings = 8192
        return config

    composed_fn = SpeculativeConfig.compose_draft_hf_overrides(custom_fn)
    assert callable(composed_fn)

    updated_config = composed_fn(hf_config)
    assert updated_config.max_position_embeddings == 8192


def test_compose_draft_hf_overrides_independent_draft_keeps_own_config():
    """Target dict overrides must not raise an independent draft's context.

    A draft loaded from its own checkpoint has its own trained context
    limit and rope configuration; replaying the target's dict overrides
    on it (apply_dict_overrides=False for independent drafts) must leave
    them untouched.
    """
    hf_config = PretrainedConfig()
    hf_config.architectures = ["MockDraftForCausalLM"]
    hf_config.max_position_embeddings = 2048

    dict_overrides = {
        "max_position_embeddings": 8192,
        "rope_parameters": {"factor": 1.5625, "rope_type": "yarn"},
    }

    composed_fn = SpeculativeConfig.compose_draft_hf_overrides(
        dict_overrides, apply_dict_overrides=False
    )
    updated_config = composed_fn(hf_config)
    assert updated_config.max_position_embeddings == 2048


def _spec(model, target_model, model_weights=None):
    return SimpleNamespace(
        model=model,
        target_model_config=SimpleNamespace(
            model=target_model, model_weights=model_weights
        ),
    )


def test_draft_shares_target_config_predicate():
    # Same-checkpoint draft (e.g. an MTP head): spec.model was defaulted to
    # the target's checkpoint, so dict overrides must be replayed on it.
    assert SpeculativeConfig._draft_shares_target_config(
        _spec("/models/qwen3", "/models/qwen3")
    )
    assert SpeculativeConfig._draft_shares_target_config(
        _spec("/models/qwen3/", "/models/qwen3")
    )
    # Weight-source aliasing (e.g. runai_streamer local cache dir).
    assert SpeculativeConfig._draft_shares_target_config(
        _spec("/cache/qwen3", "Qwen/Qwen3", model_weights="/cache/qwen3")
    )
    # Independent draft checkpoint: keeps its own config.
    assert not SpeculativeConfig._draft_shares_target_config(
        _spec("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "/models/qwen3")
    )
    assert not SpeculativeConfig._draft_shares_target_config(
        _spec(None, "/models/qwen3")
    )
