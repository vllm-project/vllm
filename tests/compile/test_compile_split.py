# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end test for vLLM compilation with VLLM_MLA_EXPOSED_SPLIT.

Tests that DeepSeek-V2-Lite can be compiled and run correctly with
VLLM_COMPILE mode and piecewise CUDA graph capture.

Example commands::

    # Run all parametrized variants (exposed_split=0 and exposed_split=1):
    pytest tests/compile/test_compile_split.py -v

    # Run only the exposed-split variant:
    pytest tests/compile/test_compile_split.py -v -k "mla_exposed_split-1"

    # Run only the non-exposed-split variant:
    pytest tests/compile/test_compile_split.py -v -k "mla_exposed_split-0"

    # Run with extra logging for debugging compilation issues:
    VLLM_LOGGING_LEVEL=DEBUG pytest tests/compile/test_compile_split.py -v -s
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
)

MODEL = "deepseek-ai/DeepSeek-V2-Lite"
PROMPTS = ["The capital of France is"]


@pytest.mark.parametrize(
    "mla_exposed_split",
    ["0", "1"],
    ids=["mla_exposed_split-0", "mla_exposed_split-1"],
)
def test_compile_split_deepseek(monkeypatch, mla_exposed_split):
    """
    Test that DeepSeek-V2-Lite compiles and generates correctly with
    VLLM_COMPILE mode, with and without VLLM_MLA_EXPOSED_SPLIT.
    """
    monkeypatch.setenv("VLLM_MLA_EXPOSED_SPLIT", mla_exposed_split)

    llm = LLM(
        model=MODEL,
        max_model_len=256,
        trust_remote_code=True,
        disable_log_stats=True,
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_inductor_graph_partition=True,
        ),
    )

    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=20, temperature=0))
    print(outputs)
    assert len(outputs) == len(PROMPTS)
    for output in outputs:
        assert len(output.outputs) > 0
        assert len(output.outputs[0].text) > 0
