# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test FP8 Marlin with MLA (Multi-head Latent Attention) models.

When FP8 Marlin repacks weights into int32, the kv_b_proj dtype detection
in the MLA attention layer must fall back to params_dtype instead of using
the non-floating-point weight dtype.

Run `pytest tests/entrypoints/llm/test_fp8_mla.py -v`.
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="FP8 Marlin is only supported on CUDA.",
)
def test_fp8_marlin_mla_generation(monkeypatch):
    monkeypatch.setenv("VLLM_TEST_FORCE_FP8_MARLIN", "1")

    llm = LLM(
        model=MODEL,
        quantization="fp8",
        enforce_eager=True,
        max_model_len=2048,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.90,
    )

    # Prompt longer than max_num_batched_tokens (512) to trigger chunked prefill.
    prompt = "Hello world, this is a test. " * 100

    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=16, temperature=0),
    )
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
