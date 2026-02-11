# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

MODELS = [
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",  # with g_idx
    "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",  # without g_idx
]
DTYPE = ["bfloat16"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_cpu_quant(vllm_runner, model, dtype):
    with vllm_runner(model, dtype=dtype) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)
    assert output
    print(output)
