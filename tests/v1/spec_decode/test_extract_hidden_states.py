# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, PreTrainedConfig

from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
)
from vllm.transformers_utils.config import get_hf_text_config
from vllm.transformers_utils.configs.extract_hidden_states import (
    ExtractHiddenStatesConfig,
)

model_dir = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class _DummyVLMConfig(PreTrainedConfig):
    """Minimal composite config that mimics VLMs like Kimi-K2.5 or LLaVA.

    The text model's parameters (hidden_size, num_attention_heads, …) live
    exclusively under `text_config`; the top-level config has none of them.
    """

    model_type = "test_vlm"

    def __init__(self, text_config: PreTrainedConfig, **kwargs):
        self.text_config = text_config
        super().__init__(architectures=["LlamaForCausalLM"], **kwargs)

    def get_text_config(self, decoder: bool = False) -> PreTrainedConfig:
        del decoder
        return self.text_config


def test_extract_hidden_states_text_only_config_regression():
    """Text-only models (no nested text_config) must keep working."""
    model_config = ModelConfig(model=model_dir, runner="generate", max_model_len=100)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        method="extract_hidden_states",
        num_speculative_tokens=1,
        draft_model_config={
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4],
            }
        },
    )

    assert speculative_config.draft_model_config is not None
    # For text-only models, hf_text_config should be the config itself.
    assert speculative_config.draft_model_config.hf_text_config is (
        speculative_config.draft_model_config.hf_config
    )
    assert (
        speculative_config.draft_model_config.hf_text_config.num_attention_heads
        == model_config.hf_text_config.num_attention_heads
    )


def test_extract_hidden_states_config_preserves_vlm_text_config():
    """A real VLM config (LLaVA) with nested text_config must be preserved."""
    text_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
    )
    vlm_config = LlavaConfig(
        vision_config=CLIPVisionConfig(),
        text_config=text_config,
    )

    # Precondition: to_dict() flattens the nested config to a plain dict.
    assert isinstance(vlm_config.to_dict()["text_config"], dict)

    extract_config = ExtractHiddenStatesConfig(
        vlm_config,
        eagle_aux_hidden_state_layer_ids=[1, 2],
    )

    # The fix: text_config is still a PreTrainedConfig, not a dict.
    assert isinstance(extract_config.text_config, LlamaConfig)

    extracted = get_hf_text_config(extract_config)
    assert extracted is extract_config.text_config
    assert extracted.num_attention_heads == text_config.num_attention_heads
    assert extracted.hidden_size == text_config.hidden_size

    # Serialization must still round-trip correctly.
    serialized = extract_config.to_dict()
    assert isinstance(serialized["text_config"], dict)
    assert serialized["text_config"]["num_attention_heads"] == (
        text_config.num_attention_heads
    )

    json_str = json.loads(extract_config.to_json_string())
    assert json_str["text_config"]["num_attention_heads"] == (
        text_config.num_attention_heads
    )


def test_extract_hidden_states_speculative_config_vlm():
    """SpeculativeConfig with a VLM target must build without errors."""
    nested_text_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
    )

    target_model_config = ModelConfig(
        model=model_dir,
        runner="generate",
        max_model_len=100,
    )
    # Replace the real text-only config with our composite VLM config.
    target_model_config.hf_config = _DummyVLMConfig(
        text_config=nested_text_config,
    )
    target_model_config.hf_text_config = nested_text_config

    speculative_config = SpeculativeConfig(
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        method="extract_hidden_states",
        num_speculative_tokens=1,
        draft_model_config={
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [1, 2],
            }
        },
    )

    assert speculative_config.draft_model_config is not None
    assert isinstance(
        speculative_config.draft_model_config.hf_config.text_config,
        LlamaConfig,
    )
    assert speculative_config.draft_model_config.hf_text_config is (
        speculative_config.draft_model_config.hf_config.text_config
    )
    assert (
        speculative_config.draft_model_config.hf_text_config.num_attention_heads
        == nested_text_config.num_attention_heads
    )


def test_extract_hidden_states_config_invalid_text_config():
    """A nested text_config missing required attrs must still be rejected."""
    broken_text_config = PreTrainedConfig(hidden_size=128)
    vlm_config = _DummyVLMConfig(text_config=broken_text_config)

    extract_config = ExtractHiddenStatesConfig(
        vlm_config,
        eagle_aux_hidden_state_layer_ids=[1],
    )

    # The object is preserved (not flattened), …
    assert extract_config.text_config is broken_text_config
    # … but validation still rejects the missing attribute.
    with pytest.raises(ValueError, match="num_attention_heads"):
        get_hf_text_config(extract_config)
