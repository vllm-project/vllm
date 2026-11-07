# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for parallel sampling (n>1) across output_kinds and engine interfaces.

Part of https://github.com/vllm-project/vllm/issues/21948
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

MODEL = "facebook/opt-125m"
PROMPTS = [
    "The capital of France is",
    "Machine learning is",
]


# ==================== AsyncLLM Tests ====================


@pytest.mark.parametrize(
    "output_kind",
    [
        RequestOutputKind.CUMULATIVE,
        RequestOutputKind.DELTA,
        RequestOutputKind.FINAL_ONLY,
    ],
)
@pytest.mark.asyncio
async def test_async_llm_parallel_sampling(output_kind):
    """Test AsyncLLM.generate with n>1 across all output_kinds."""
    n = 3
    max_tokens = 20

    engine_args = AsyncEngineArgs(model=MODEL, enforce_eager=True)
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=1.0,
        output_kind=output_kind,
    )

    final_output = None
    async for out in engine.generate(
        request_id="test-parallel-0",
        prompt=PROMPTS[0],
        sampling_params=sampling_params,
    ):
        final_output = out

    assert final_output is not None, "No output received from AsyncLLM.generate"
    assert len(final_output.outputs) == n, (
        f"Expected {n} completions, got {len(final_output.outputs)}"
    )

    # Verify each completion has valid content
    for i, comp in enumerate(final_output.outputs):
        assert comp.index == i, f"Expected index {i}, got {comp.index}"
        assert len(comp.token_ids) > 0, f"Completion {i} has no tokens"
        if output_kind != RequestOutputKind.DELTA:
            assert len(comp.text) > 0, f"Completion {i} has no text"

    # Verify completions are diverse (not all identical)
    texts = {comp.text for comp in final_output.outputs}
    # With temperature=1.0 and n=3, expect at least 2 unique
    assert len(texts) >= 2, f"Expected diverse completions with n={n}, but got: {texts}"

    engine.shutdown()


# ==================== LLM.generate Tests ====================


@pytest.mark.parametrize(
    "output_kind",
    [
        RequestOutputKind.CUMULATIVE,
        RequestOutputKind.DELTA,
        RequestOutputKind.FINAL_ONLY,
    ],
)
def test_llm_parallel_sampling_output_kinds(output_kind):
    """Test LLM.generate with n>1 across all output_kinds."""
    n = 3
    max_tokens = 20

    llm = LLM(model=MODEL, enforce_eager=True)
    sampling_params = SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=1.0,
        output_kind=output_kind,
    )

    outputs = llm.generate(PROMPTS, sampling_params)

    for prompt_idx, out in enumerate(outputs):
        assert len(out.outputs) == n, (
            f"Prompt {prompt_idx}: expected {n} completions, got {len(out.outputs)}"
        )
        for i, comp in enumerate(out.outputs):
            assert comp.index == i, (
                f"Prompt {prompt_idx}: expected index {i}, got {comp.index}"
            )
            assert len(comp.token_ids) > 0, (
                f"Prompt {prompt_idx}, completion {i}: no tokens generated"
            )
            assert comp.finish_reason is not None, (
                f"Prompt {prompt_idx}, completion {i}: missing finish_reason"
            )


# ==================== Streaming Delta Validation ====================


@pytest.mark.asyncio
async def test_async_llm_parallel_sampling_delta_streaming():
    """Test that DELTA output_kind correctly streams incremental tokens."""
    n = 2
    max_tokens = 15

    engine_args = AsyncEngineArgs(model=MODEL, enforce_eager=True)
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=1.0,
        output_kind=RequestOutputKind.DELTA,
    )

    # Collect all deltas per completion index
    all_token_ids: dict[int, list] = {i: [] for i in range(n)}
    async for out in engine.generate(
        request_id="test-delta-stream",
        prompt=PROMPTS[0],
        sampling_params=sampling_params,
    ):
        for comp in out.outputs:
            all_token_ids[comp.index].extend(comp.token_ids)

    # Each completion should have accumulated tokens up to max_tokens
    for idx in range(n):
        total_tokens = len(all_token_ids[idx])
        assert total_tokens > 0, f"Completion {idx}: no delta tokens received"
        assert total_tokens <= max_tokens, (
            f"Completion {idx}: got {total_tokens} tokens, expected <= {max_tokens}"
        )

    engine.shutdown()


# ==================== Cumulative Monotonicity ====================


@pytest.mark.asyncio
async def test_async_llm_parallel_sampling_cumulative_monotonic():
    """Test that CUMULATIVE output grows monotonically in length."""
    n = 2
    max_tokens = 15

    engine_args = AsyncEngineArgs(model=MODEL, enforce_eager=True)
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=1.0,
        output_kind=RequestOutputKind.CUMULATIVE,
    )

    prev_lengths: dict[int, int] = {i: 0 for i in range(n)}
    async for out in engine.generate(
        request_id="test-cumulative",
        prompt=PROMPTS[0],
        sampling_params=sampling_params,
    ):
        for comp in out.outputs:
            current_len = len(comp.token_ids)
            assert current_len >= prev_lengths[comp.index], (
                f"Completion {comp.index}: cumulative length decreased "
                f"from {prev_lengths[comp.index]} to {current_len}"
            )
            prev_lengths[comp.index] = current_len

    engine.shutdown()
