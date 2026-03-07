# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PretrainedConfig

from vllm.config.model import _get_and_verify_max_len
from vllm.config.model_arch import ModelArchitectureConfig


def _make_model_arch_config(
    derived_max_model_len: int = 4096,
) -> ModelArchitectureConfig:
    return ModelArchitectureConfig(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        text_model_type=None,
        hidden_size=4096,
        total_num_hidden_layers=32,
        total_num_attention_heads=32,
        head_size=128,
        vocab_size=32000,
        total_num_kv_heads=32,
        num_experts=0,
        quantization_config=None,
        is_deepseek_mla=False,
        derived_max_model_len_and_key=(
            derived_max_model_len,
            "max_position_embeddings",
        ),
    )


def test_get_and_verify_max_len_handles_null_rope_factor():
    hf_config = PretrainedConfig(model_type="llama")
    hf_config.rope_parameters = {
        "rope_type": "default",
        "factor": None,
    }

    max_model_len = _get_and_verify_max_len(
        hf_config=hf_config,
        model_arch_config=_make_model_arch_config(4096),
        tokenizer_config=None,
        max_model_len=None,
        disable_sliding_window=False,
        sliding_window=None,
    )

    assert max_model_len == 4096


def test_get_and_verify_max_len_still_applies_valid_rope_factor():
    hf_config = PretrainedConfig(model_type="llama")
    hf_config.rope_parameters = {
        "rope_type": "linear",
        "factor": 2.0,
    }

    max_model_len = _get_and_verify_max_len(
        hf_config=hf_config,
        model_arch_config=_make_model_arch_config(4096),
        tokenizer_config=None,
        max_model_len=None,
        disable_sliding_window=False,
        sliding_window=None,
    )

    assert max_model_len == 8192
