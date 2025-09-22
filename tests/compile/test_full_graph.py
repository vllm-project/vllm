# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import tempfile
from typing import Any, Optional, Union

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from tests.v1.attention.utils import _Backend
from vllm import LLM, SamplingParams
from vllm.attention.selector import global_force_attn_backend_context_manager
from vllm.config import (CompilationConfig, CompilationLevel, CUDAGraphMode,
                         PassConfig)
from vllm.platforms import current_platform
from vllm.utils import is_torch_equal_or_newer

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

    with monkeypatch.context():
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
    ] + [
        # Test depyf integration works
        (CompilationConfig(level=CompilationLevel.PIECEWISE,
                           debug_dump_path=tempfile.gettempdir()),
         ("facebook/opt-125m", {})),
    ] + [
        # graph inductor partition
        (
            CompilationConfig(
                level=CompilationLevel.PIECEWISE,
                # inductor graph partition uses
                # torch._C.Tag.cudagraph_unsafe to specify splitting ops
                use_inductor_graph_partition=True,
                cudagraph_mode=CUDAGraphMode.PIECEWISE,
                compile_sizes=[1, 2]),
            model) for model in models_list(all=False)
        if is_torch_equal_or_newer("2.9.0.dev")
    ])
# only test some of the models
@create_new_process_for_each_test()
def test_custom_compile_config(
    compilation_config: CompilationConfig,
    model_info: tuple[str, dict[str, Any]],
):
    if (compilation_config.use_inductor_graph_partition
            and not is_torch_equal_or_newer("2.9.0.dev")):
        pytest.skip("inductor graph partition is only available "
                    "in PyTorch 2.9+")

    model, model_kwargs = model_info
    print(f"MODEL={model}")
    run_model(compilation_config, model, model_kwargs)


def test_inductor_graph_partition_attn_fusion(caplog_vllm):
    if not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("inductor graph partition is only available "
                    "in PyTorch 2.9+")

    model = "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8"
    compilation_config = CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        use_inductor_graph_partition=True,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        custom_ops=["+quant_fp8"],
        pass_config=PassConfig(enable_attn_fusion=True, enable_noop=True),
    )
    model_kwargs = {
        "kv_cache_dtype": "fp8",
        "max_model_len": 1024,
    }
    with caplog_vllm.at_level(
            logging.DEBUG), global_force_attn_backend_context_manager(
                _Backend.FLASHINFER):
        run_model(compilation_config, model, model_kwargs)

    try:
        assert ("Fused quantization onto 48 attention nodes"
                in caplog_vllm.text), caplog_vllm.text
    except AssertionError:
        # Note: this message is only triggered when the compilation goes
        # through the custom pass. Due to multiple layers of cache on
        # PyTorch side, the compilation of a graph may be cached such
        # that custom pass directly goes through cache. In this case,
        # we go through this branch and assert that the pass is not
        # triggered.
        assert "Fused quantization" not in caplog_vllm.text


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
