# SPDX-License-Identifier: Apache-2.0
"""Tests whether Marlin models can be loaded from the autogptq config.

Run `pytest tests/quantization/test_configs.py --forked`.
"""

from dataclasses import dataclass

import pytest

from vllm.config import ModelConfig


@dataclass
class ModelPair:
    model_marlin: str
    model_gptq: str


# Model Id // Quantization Arg // Expected Type
MODEL_ARG_EXPTYPES = [
    # AUTOGPTQ
    # compat: autogptq <=0.7.1 is_marlin_format: bool
    # Model Serialized in Marlin Format should always use Marlin kernel.
    ("neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin", None, "marlin"),
    ("neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin", "marlin", "marlin"),
    ("neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin", "gptq", "marlin"),
    ("neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin", "awq", "ERROR"),
    # Model Serialized in Exllama Format.
    ("TheBloke/Llama-2-7B-Chat-GPTQ", None, "gptq_marlin"),
    ("TheBloke/Llama-2-7B-Chat-GPTQ", "marlin", "gptq_marlin"),
    ("TheBloke/Llama-2-7B-Chat-GPTQ", "gptq", "gptq"),
    ("TheBloke/Llama-2-7B-Chat-GPTQ", "awq", "ERROR"),
    # compat: autogptq >=0.8.0 use checkpoint_format: str
    # Model Serialized in Marlin Format should always use Marlin kernel.
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit", None, "marlin"),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit", "marlin", "marlin"),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit", "gptq", "marlin"),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit", "awq", "ERROR"),
    # Model Serialized in Exllama Format.
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", None, "gptq_marlin"),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", "marlin", "gptq_marlin"),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", "gptq", "gptq"),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", "awq", "ERROR"),

    # AUTOAWQ
    ("TheBloke/OpenHermes-2.5-Mistral-7B-AWQ", None, "awq_marlin"),
    ("TheBloke/OpenHermes-2.5-Mistral-7B-AWQ", "awq", "awq"),
    ("TheBloke/OpenHermes-2.5-Mistral-7B-AWQ", "marlin", "awq_marlin"),
    ("TheBloke/OpenHermes-2.5-Mistral-7B-AWQ", "gptq", "ERROR"),
]


@pytest.mark.parametrize("model_arg_exptype", MODEL_ARG_EXPTYPES)
def test_auto_gptq(model_arg_exptype: tuple[str, None, str]) -> None:
    model_path, quantization_arg, expected_type = model_arg_exptype

    try:
        model_config = ModelConfig(model_path,
                                   task="auto",
                                   tokenizer=model_path,
                                   tokenizer_mode="auto",
                                   trust_remote_code=False,
                                   seed=0,
                                   dtype="float16",
                                   revision=None,
                                   quantization=quantization_arg)
        found_quantization_type = model_config.quantization
    except ValueError:
        found_quantization_type = "ERROR"

    assert found_quantization_type == expected_type, (
        f"Expected quant_type == {expected_type} for {model_path}, "
        f"but found {found_quantization_type} "
        f"for no --quantization {quantization_arg} case")
