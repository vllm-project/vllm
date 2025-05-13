# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional, Union

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel, PassConfig
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test


def models_list(*, all: bool = True, keywords: Optional[list[str]] = None):
    TEST_MODELS: list[tuple[str, dict[str, Any]]] = [
        ("facebook/opt-125m", {}),
        ("nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change", {
            "dtype": torch.float16,
        }),
        ("neuralmagic/Llama-3.2-1B-Instruct-FP8-dynamic", {
            "dtype": torch.float16,
        }),
        ("neuralmagic/Llama-3.2-1B-Instruct-quantized.w8a8", {}),
        ("meta-llama/Llama-3.2-1B-Instruct", {}),
    ]

    if all:
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

    if keywords is None:
        return TEST_MODELS

    # filter by keywords
    pred = lambda model: any(keyword in model[0] for keyword in keywords)
    return list(filter(pred, TEST_MODELS))


@pytest.mark.parametrize(
    "optimization_level",
    [CompilationLevel.DYNAMO_ONCE, CompilationLevel.PIECEWISE],
)
@pytest.mark.parametrize("model_info", models_list(all=True))
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

        run_model(optimization_level, model, model_kwargs)


# TODO(luka) add other supported compilation config scenarios here
@pytest.mark.parametrize(
    "compilation_config, model_info",
    [
        # additional compile sizes, only some of the models
        (CompilationConfig(level=CompilationLevel.PIECEWISE,
                           compile_sizes=[1, 2]), model)
        for model in models_list(all=False)
    ] + [
        # RMSNorm + quant fusion, only 8-bit quant models
        (CompilationConfig(level=CompilationLevel.PIECEWISE,
                           custom_ops=["+rms_norm"],
                           pass_config=PassConfig(enable_fusion=True,
                                                  enable_noop=True)), model)
        for model in models_list(keywords=["FP8-dynamic", "quantized.w8a8"])
    ])
# only test some of the models
@create_new_process_for_each_test()
def test_custom_compile_config(
    compilation_config: CompilationConfig,
    model_info: tuple[str, dict[str, Any]],
):
    model, model_kwargs = model_info
    print(f"MODEL={model}")
    run_model(compilation_config, model, model_kwargs)


def run_model(compile_config: Union[int, CompilationConfig], model: str,
              model_kwargs: dict[str, Any]):
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
        compilation_config=compile_config,
        **model_kwargs,
    )
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
