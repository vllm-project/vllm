# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams
from vllm.config import CompilationLevel
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test


@pytest.fixture(params=None, name="model_info")
def models_list_fixture(request):
    TEST_MODELS: list[tuple[str, dict[str, Any]]] = [
        ("facebook/opt-125m", {}),
        ("nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change", {
            "dtype": torch.float16,
            "quantization": "compressed-tensors"
        }),
        ("neuralmagic/Llama-3.2-1B-Instruct-FP8-dynamic", {
            "dtype": torch.float16,
            "quantization": "compressed-tensors"
        }),
        ("neuralmagic/Llama-3.2-1B-Instruct-quantized.w8a8", {
            "quantization": "compressed-tensors"
        }),
        ("meta-llama/Llama-3.2-1B-Instruct", {}),
    ]

    if is_quant_method_supported("aqlm"):
        TEST_MODELS.append(("ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf", {
            "quantization": "aqlm"
        }))

    # TODO: figure out why this fails.
    if False and is_quant_method_supported("gguf"):  # noqa: SIM223
        TEST_MODELS.append(("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", {
            "quantization": "gguf"
        }))

    if is_quant_method_supported("gptq"):
        TEST_MODELS.append(("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ", {
            "quantization": "gptq"
        }))

    if is_quant_method_supported("gptq_marlin"):
        TEST_MODELS.append(("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", {
            "quantization": "gptq_marlin"
        }))

    if is_quant_method_supported("gptq_marlin_24"):
        TEST_MODELS.append(("alexm-nm/tinyllama-24-marlin24-4bit-g128", {
            "quantization": "gptq_marlin_24"
        }))

    if is_quant_method_supported("marlin"):
        TEST_MODELS.append(
            ("robertgshaw2/TinyLlama-1.1B-Chat-v1.0-g128-marlin", {
                "quantization": "marlin"
            }))

    if not current_platform.is_rocm() and is_quant_method_supported("awq"):
        TEST_MODELS.append(("TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ", {
            "quantization": "AWQ"
        }))

    return TEST_MODELS


@pytest.mark.parametrize(
    "optimization_level",
    [CompilationLevel.DYNAMO_ONCE, CompilationLevel.PIECEWISE],
)
@pytest.mark.parametrize("model_info", "", indirect=True)
@create_new_process_for_each_test()
def test_full_graph(
    monkeypatch: pytest.MonkeyPatch,
    model_info: tuple[str, dict[str, Any]],
    optimization_level: int,
):
    model, model_kwargs = model_info

    with monkeypatch.context() as m:
        # make sure these models can be captured in full graph mode
        m.setenv("VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1")
        print(f"MODEL={model}")

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(temperature=0)
        llm = LLM(
            model=model,
            enforce_eager=True,
            tensor_parallel_size=1,
            disable_custom_all_reduce=True,
            compilation_config=optimization_level,
            **model_kwargs,
        )
        outputs = llm.generate(prompts, sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
