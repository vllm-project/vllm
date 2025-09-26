# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata
import importlib.util

import pytest
import torch

DTYPE = ["bfloat16"]

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_pre_quantized_model(vllm_runner):
    with vllm_runner("drisspg/fp8-opt-125m",
                     quantization="torchao",
                     dtype="bfloat16",
                     enforce_eager=True) as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)
    assert output
    print(output)


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.parametrize(
    "pt_load_map_location",
    [
        "cuda:0",
        # {"": "cuda"},
    ])
def test_opt_125m_int8wo_model_loading_with_params(vllm_runner,
                                                   pt_load_map_location):
    torch._dynamo.reset()
    model_name = "jerryzh168/opt-125m-int8wo-partial-quant"
    with vllm_runner(model_name=model_name,
                     quantization="torchao",
                     dtype="bfloat16",
                     pt_load_map_location=pt_load_map_location) as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)

        assert output
        print(output)


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_opt_125m_int4wo_model_per_module_quant(vllm_runner):
    torch._dynamo.reset()
    model_name = "jerryzh168/opt-125m-int4wo-per-module"
    with vllm_runner(model_name=model_name,
                     quantization="torchao",
                     dtype="bfloat16",
                     pt_load_map_location="cuda:0") as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)

        assert output
        print(output)


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_qwenvl_int8wo_model_loading_with_params(vllm_runner):
    torch._dynamo.reset()
    model_name = "mobicham/Qwen2.5-VL-3B-Instruct_int8wo_ao"
    with vllm_runner(model_name=model_name,
                     quantization="torchao",
                     dtype="bfloat16",
                     pt_load_map_location="cuda:0") as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)

        assert output
        print(output)


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
@pytest.mark.skip(
    reason="since torchao nightly is only compatible with torch nightly"
    "currently https://github.com/pytorch/ao/issues/2919, we'll have to skip "
    "torchao tests that requires newer versions (0.14.0.dev+) for now")
def test_opt_125m_awq_int4wo_model_loading_with_params(vllm_runner):
    torch._dynamo.reset()
    model_name = ("torchao-testing/opt-125m-AWQConfig-Int4WeightOnlyConfig-v2"
                  "-0.14.0.dev")
    with vllm_runner(model_name=model_name,
                     quantization="torchao",
                     dtype="bfloat16",
                     pt_load_map_location="cuda:0") as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)

        assert output
        print(output)


if __name__ == "__main__":
    pytest.main([__file__])
