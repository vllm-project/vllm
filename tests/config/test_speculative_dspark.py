# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.model_executor.layers import quantization as me_quant
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


class _NoOverrideQuantizationConfig:
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        return None


@pytest.mark.cpu_test
def test_deepseek_v4_dspark_preserves_specialized_fp8_quantization(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    quantization_configs = {
        "fp8": Fp8Config,
        "deepseek_v4_fp8": DeepseekV4FP8Config,
    }
    # Keep this config test independent of unrelated optional quant backends.
    monkeypatch.setattr(me_quant, "QUANTIZATION_METHODS", list(quantization_configs))
    monkeypatch.setattr(
        me_quant,
        "get_quantization_config",
        lambda name: quantization_configs.get(name, _NoOverrideQuantizationConfig),
    )

    model_path = tmp_path / "deepseek-v4-dspark"
    DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=32,
        torch_dtype="bfloat16",
        quantization_config={
            "activation_scheme": "dynamic",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    ).save_pretrained(model_path)
    target_model_config = ModelConfig(
        model=str(model_path), tokenizer_mode="skip", max_model_len=32
    )

    speculative_config = SpeculativeConfig(
        model=str(model_path),
        method="dspark",
        num_speculative_tokens=5,
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
    )

    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config.architectures == ["DSparkDraftModel"]
    assert draft_model_config.quantization == "deepseek_v4_fp8"
