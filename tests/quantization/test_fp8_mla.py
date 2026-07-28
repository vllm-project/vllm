# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test FP8 Marlin with MLA (Multi-head Latent Attention) models.

When FP8 Marlin repacks weights into int32, the kv_b_proj dtype detection
in the MLA attention layer must fall back to params_dtype instead of using
the non-floating-point weight dtype.

Run `pytest tests/quantization/test_fp8_mla.py -v`.
"""

import pytest

from tests.quantization.utils import is_quant_method_supported

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_fp8_marlin_mla_generation(vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_TEST_FORCE_FP8_MARLIN", "1")

    with vllm_runner(
        MODEL,
        quantization="fp8",
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
    ) as llm:
        sched_config = llm.llm.llm_engine.vllm_config.scheduler_config
        assert sched_config.enable_chunked_prefill, (
            "Chunked prefill must be enabled for this test"
        )

        # Prompt longer than max_num_batched_tokens to trigger chunked prefill.
        prompt = "Hello world, this is a test. " * 500
        tokenizer = llm.llm.get_tokenizer()
        num_tokens = len(tokenizer.encode(prompt))
        assert num_tokens > sched_config.max_num_batched_tokens, (
            f"Prompt ({num_tokens} tokens) must exceed "
            f"max_num_batched_tokens ({sched_config.max_num_batched_tokens}) "
            f"to trigger chunked prefill"
        )

        outputs = llm.generate_greedy([prompt], max_tokens=16)
        assert len(outputs) == 1
        assert outputs[0][1]
