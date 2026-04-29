# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for ``thinking_token_budget`` with reasoning models.

Covers Qwen3-0.6B and Qwen3.5 FP8 + MTP.
"""

import asyncio
import json
from typing import Literal

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer, multi_gpu_only, requires_fp8
from vllm.platforms import current_platform
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
QWEN35_FP8_MTP_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
MESSAGES = [{"role": "user", "content": "What is 1+1? Be concise."}]
THINK_BUDGET = 5

REASONING_START_STR = "<think>"
REASONING_END_STR = "</think>"


def _count_reasoning_decode_token_ids_between_markers(
    full_token_ids: list[int],
    reasoning_start_ids: list[int],
    reasoning_end_ids: list[int],
) -> int | None:
    """Count decode tokens in the thinking span (after last start, before first end)."""

    if not reasoning_start_ids or not reasoning_end_ids:
        raise ValueError("reasoning marker token id lists must be non-empty")

    def _last_subseq_index(haystack: list[int], needle: list[int]) -> int:
        n = len(needle)
        if n > len(haystack):
            return -1
        for i in range(len(haystack) - n, -1, -1):
            if haystack[i : i + n] == needle:
                return i
        return -1

    last_start = _last_subseq_index(full_token_ids, reasoning_start_ids)
    if last_start < 0:
        return None

    pos_after_start = last_start + len(reasoning_start_ids)
    end_n = len(reasoning_end_ids)
    for j in range(pos_after_start, len(full_token_ids) - end_n + 1):
        if full_token_ids[j : j + end_n] == reasoning_end_ids:
            return j - pos_after_start
    return len(full_token_ids) - pos_after_start


@pytest.fixture(scope="module")
def server():
    args = [
        "--reasoning-parser",
        "qwen3",
        "--reasoning-config",
        '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}',
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
        "--no-async-scheduling",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_with_auto_reasoning_config():
    args = [
        "--reasoning-parser",
        "qwen3",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
        "--no-async-scheduling",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_qwen35_fp8_mtp_tp2():
    """Qwen3.5-35B FP8 with MTP speculative decoding and tensor parallel size 2."""
    if current_platform.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for --tensor-parallel-size 2")
    if not current_platform.supports_fp8():
        pytest.skip("FP8 is not supported on this platform")

    spec_cfg = {
        "method": "mtp",
        "num_speculative_tokens": 2,
        "max_model_len": 32768,
    }
    args = [
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "32768",
        "--speculative-config",
        json.dumps(spec_cfg),
        "--reasoning-parser",
        "qwen3",
        "--reasoning-config",
        json.dumps(
            {
                "reasoning_start_str": REASONING_START_STR,
                "reasoning_end_str": REASONING_END_STR,
            }
        ),
    ]
    # With 4+ GPUs, run TP=2 on physical devices 2,3 so module-scoped 0.6B servers
    # on 0,1 do not exhaust memory on the same devices as this worker.
    env_dict = None
    if current_platform.device_count() >= 4:
        env_dict = {"CUDA_VISIBLE_DEVICES": "2,3"}

    with RemoteOpenAIServer(
        QWEN35_FP8_MTP_MODEL,
        args,
        max_wait_seconds=3000,
        env_dict=env_dict,
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(request, server, server_with_auto_reasoning_config):
    server_map = {
        "default": server,
        "auto_config": server_with_auto_reasoning_config,
    }
    target_server = server_map[request.param]
    async with target_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["default", "auto_config"], indirect=True)
async def test_thinking_token_budget_mixed_requests(client: openai.AsyncOpenAI):
    """Test that mixed requests (some with thinking_token_budget, some without)
    complete successfully without errors."""

    response_with_budget = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=100,
        extra_body={"thinking_token_budget": THINK_BUDGET},
    )
    response_without_budget = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=100,
    )

    msg_with = response_with_budget.choices[0].message
    msg_without = response_without_budget.choices[0].message

    assert msg_with.content or getattr(msg_with, "reasoning", None)
    assert msg_without.content or getattr(msg_without, "reasoning", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["default", "auto_config"], indirect=True)
async def test_thinking_token_budget_limits_reasoning(client: openai.AsyncOpenAI):
    """Test that thinking_token_budget limits the number of reasoning tokens.

    Counts non-empty streaming ``delta.reasoning`` chunks (coarse proxy; each
    chunk may represent multiple decode tokens — see
    ``_count_reasoning_decode_token_ids_between_markers`` and the Qwen3.5 MTP
    test for id-based checks).
    """

    reasoning_token_count = 0
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=100,
        stream=True,
        extra_body={"thinking_token_budget": THINK_BUDGET},
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning", None):
            reasoning_token_count += 1

    assert reasoning_token_count == THINK_BUDGET, (
        f"reasoning tokens ({reasoning_token_count}) exceeded "
        f"thinking_token_budget ({THINK_BUDGET})"
    )


@pytest.mark.asyncio
@multi_gpu_only(num_gpus=2)
@requires_fp8
async def test_thinking_token_budget_qwen35_fp8_mtp_concurrent_mixed_budget_and_plain(
    server_qwen35_fp8_mtp_tp2,
):
    """Concurrent chat requests: some with ``thinking_token_budget``, some without.

    Exercises the scheduler / input processor under a mixed batch on the same
    Qwen3.5 FP8 + MTP (TP=2) server. Budgeted calls are checked with
    ``_count_reasoning_decode_token_ids_between_markers`` on full token ids.
    """

    _batch_spec: list[tuple[Literal["budget"], int] | tuple[Literal["plain"], None]] = [
        ("budget", 1),
        ("budget", 12),
        ("plain", None),
        ("budget", 20),
        ("budget", 14),
        ("plain", None),
        ("plain", None),
        ("budget", 12),
        ("plain", None),
    ]

    tokenizer = get_tokenizer(tokenizer_name=QWEN35_FP8_MTP_MODEL)
    start_ids = list(tokenizer.encode(REASONING_START_STR, add_special_tokens=False))
    end_ids = list(tokenizer.encode(REASONING_END_STR, add_special_tokens=False))

    async with server_qwen35_fp8_mtp_tp2.get_async_client() as client:

        async def budgeted_call(expected_budget: int):
            return await client.chat.completions.create(
                model=QWEN35_FP8_MTP_MODEL,
                messages=MESSAGES,
                max_tokens=256,
                stream=False,
                extra_body={
                    "thinking_token_budget": expected_budget,
                    "return_token_ids": True,
                },
            )

        async def plain_call():
            return await client.chat.completions.create(
                model=QWEN35_FP8_MTP_MODEL,
                messages=MESSAGES,
                max_tokens=256,
                stream=False,
            )

        coros = []
        for row in _batch_spec:
            if row[0] == "budget":
                b = row[1]
                assert isinstance(b, int)
                coros.append(budgeted_call(b))
            else:
                coros.append(plain_call())
        results = await asyncio.gather(*coros)

    for i, (response, (kind, expected_budget)) in enumerate(
        zip(results, _batch_spec, strict=True)
    ):
        msg = response.choices[0].message
        assert msg.content or getattr(msg, "reasoning", None), (
            f"index {i} ({kind}): empty message"
        )

        if kind == "budget":
            assert expected_budget is not None
            assert response.prompt_token_ids is not None
            assert response.choices[0].token_ids is not None
            full_ids = list(response.prompt_token_ids) + list(
                response.choices[0].token_ids
            )
            n_reason = _count_reasoning_decode_token_ids_between_markers(
                full_ids, start_ids, end_ids
            )
            assert n_reason is not None, f"index {i}: missing reasoning start in ids"
            assert n_reason == expected_budget, (
                f"index {i}: reasoning decode token ids ({n_reason}) != "
                f"thinking_token_budget ({expected_budget})"
            )
