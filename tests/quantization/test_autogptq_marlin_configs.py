"""Tests whether Marlin models can be loaded from the autogptq config.

Run `pytest tests/quantization/test_autogptq_marlin_configs.py --forked`.
"""

from dataclasses import dataclass

import pytest

from vllm.config import ModelConfig


@dataclass
class ModelPair:
    model_marlin: str
    model_gptq: str


# Model Id // Expected Kernel
MODELS_QUANT_TYPE = [
    # compat: autogptq <=0.7.1 is_marlin_format: bool
    ("neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin", "marlin"),
    ("TheBloke/Llama-2-7B-Chat-GPTQ", "gptq"),
    # compat: autogptq >=0.8.0 use checkpoint_format: str
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit", "marlin"),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", "gptq")
]


@pytest.mark.parametrize("model_quant_type", MODELS_QUANT_TYPE)
def test_auto_gptq(model_quant_type: str, ) -> None:
    model_path, quant_type = model_quant_type

    model_config_no_quant_arg = ModelConfig(
        model_path,
        model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
        quantization=None  # case 1
    )

    model_config_quant_arg = ModelConfig(
        model_path,
        model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
        quantization="gptq"  # case 2
    )

    assert model_config_no_quant_arg.quantization == quant_type, (
        f"Expected quant_type == {quant_type} for {model_path}, "
        f"but found {model_config_no_quant_arg.quantization} "
        "for no --quantization None case")

    assert model_config_quant_arg.quantization == quant_type, (
        f"Expected quant_type == {quant_type} for {model_path}, "
        f"but found {model_config_quant_arg.quantization} "
        "for --quantization gptq case")
