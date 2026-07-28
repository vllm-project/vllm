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

MODEL = "RedHatAI/DeepSeek-Coder-V2-Instruct-FP8"


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="FP8 Marlin is only supported on CUDA.",
)
def test_fp8_marlin_mla_generation(monkeypatch):
    monkeypatch.setenv("VLLM_TEST_FORCE_FP8_MARLIN", "1")

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.90,
    )

    # Verify chunked prefill is actually enabled.
    sched_config = llm.llm_engine.vllm_config.scheduler_config
    assert sched_config.enable_chunked_prefill, (
        "Chunked prefill must be enabled for this test"
    )

    # Prompt longer than max_num_batched_tokens to trigger chunked prefill.
    prompt = "Hello world, this is a test. " * 500
    tokenizer = llm.get_tokenizer()
    num_tokens = len(tokenizer.encode(prompt))
    assert num_tokens > sched_config.max_num_batched_tokens, (
        f"Prompt ({num_tokens} tokens) must exceed "
        f"max_num_batched_tokens ({sched_config.max_num_batched_tokens}) "
        f"to trigger chunked prefill"
    )

    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=16, temperature=0),
    )
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
