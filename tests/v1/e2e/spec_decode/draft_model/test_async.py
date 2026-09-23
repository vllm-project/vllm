# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for sync-free speculative decoding with async scheduling."""

import pytest

# Test configurations: (model, spec_model, method, num_spec_tokens, backend_env)
SPEC_DECODE_CONFIGS = [
    pytest.param(
        "meta-llama/Llama-3.2-1B-Instruct",
        "nm-testing/Llama3_2_1B_speculator.eagle3",
        "eagle3",
        2,
        id="eagle3-llama",
    ),
    pytest.param(
        "eagle618/deepseek-v3-random",
        "eagle618/eagle-deepseek-v3-random",
        "eagle",
        2,
        id="eagle-mla-deepseek",
    ),
    pytest.param(
        "Qwen/Qwen3.5-0.8B-Base",
        "Qwen/Qwen3.5-0.8B-Base",
        "mtp",
        1,
        id="mtp-qwen3_5-hybrid",
    ),
]


@pytest.mark.parametrize(
    "model,spec_model,method,num_spec_tokens",
    SPEC_DECODE_CONFIGS,
)
def test_no_sync_with_spec_decode(
    model: str,
    spec_model: str,
    method: str,
    num_spec_tokens: int,
    monkeypatch,
    vllm_runner,
):
    """Test generation while the worker rejects unintended GPU-CPU syncs."""
    monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", "error")

    from vllm import SamplingParams
    from vllm.config import CompilationConfig

    # Qwen3.5 is a VLM; without this, profile_run runs the ViT warmup
    # and peaks well above the 18GB MIG slice used by one of the CI lanes.
    # This test only exercises text generation, so the vision tower is
    # never needed.
    extra_kwargs: dict = {}
    if "Qwen3.5" in model:
        extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    with vllm_runner(
        model,
        block_size=None,
        trust_remote_code=False,
        max_model_len=256,
        speculative_config={
            "method": method,
            "num_speculative_tokens": num_spec_tokens,
            "model": spec_model,
        },
        enforce_eager=True,
        async_scheduling=True,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
        **extra_kwargs,
    ) as runner:
        llm = runner.llm

        # Assert async scheduling is actually active before running inference.
        assert llm.llm_engine.vllm_config.scheduler_config.async_scheduling, (
            f"Expected async_scheduling=True for spec decode, got False. "
            f"method={method}, target={model}, draft={spec_model}"
        )

        outputs = llm.generate(
            ["Hello, my name is"],
            SamplingParams(temperature=0, max_tokens=10),
        )

        assert len(outputs) == 1, (
            f"{method} target={model}: expected one request output, got {len(outputs)}"
        )
        assert outputs[0].outputs, (
            f"{method} target={model}: request output has no completion candidates"
        )
        assert outputs[0].outputs[0].text, (
            f"{method} target={model}: generated completion text is empty"
        )
