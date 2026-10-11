# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

MODELS = [
    "RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8",  # INT8 W8A8 MoE
]
DTYPE = ["bfloat16"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_cpu_w8a8(vllm_runner, model, dtype):
    with vllm_runner(model, dtype=dtype) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)
    assert output
    print(output)
