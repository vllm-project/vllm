# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise bounded text sessions through the real asynchronous engine."""

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest
import regex as re

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.openai.streaming.chunking import render_incremental_turn
from vllm.renderers.params import ChatParams
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.streaming.retention import StreamingRetentionParams

if TYPE_CHECKING:
    from vllm.renderers.chat_utils import ChatCompletionMessageParam

MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.asyncio
async def test_bounded_text_turns_evict_replay_and_finish_naturally(capfd, monkeypatch):
    """Keep the latest answer and per-turn output state after repeated replay."""
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=MODEL,
            enforce_eager=True,
            async_scheduling=False,
            enable_prefix_caching=True,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=1,
            kv_cache_memory_bytes=128 * 1024**2,
            disable_log_stats=True,
        )
    )
    try:
        position_limit = engine.model_config.hf_text_config.max_position_embeddings
        retention = StreamingRetentionParams(
            max_text_tokens=64,
            max_session_tokens=256,
            reprefill_threshold=320 / position_limit,
        )
        params = SamplingParams(
            temperature=0,
            max_tokens=4,
            output_kind=RequestOutputKind.DELTA,
            logprobs=1,
            structured_outputs=StructuredOutputsParams(choice=["A", "B"]),
            extra_args={"streaming_retention": retention},
        )

        chat_params = ChatParams(
            chat_template_kwargs={
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
        )
        prompts: list[list[int]] = []
        completed: list[str] = []
        finishes: set[str] = set()
        replies: asyncio.Queue[tuple[list[int], str]] = asyncio.Queue(maxsize=1)
        producer_finished = asyncio.Event()
        turns = 64

        async def inputs() -> AsyncGenerator[StreamingInput, None]:
            replay_token = None
            for index in range(turns):
                label = "A" if index % 2 == 0 else "B"
                user: ChatCompletionMessageParam = {
                    "role": "user",
                    "content": f"Output {label}.",
                }
                if index == 0:
                    messages: list[ChatCompletionMessageParam] = [
                        {
                            "role": "system",
                            "content": (
                                "For each user turn, output exactly the letter "
                                "named in that turn. Output only A or B."
                            ),
                        },
                        user,
                    ]
                    _, (prompt,) = await engine.renderer.render_chat_async(
                        [messages], chat_params
                    )
                else:
                    prompt = await render_incremental_turn(
                        engine.renderer, [user], chat_params, replay_token
                    )
                assert prompt["type"] == "token"
                prompts.append(list(prompt["prompt_token_ids"]))
                turn_params = params.clone()
                # A one-token answer also exercises replay of a visible final
                # token that the scheduler discards at a length boundary.
                turn_params.max_tokens = 1 if index % 2 == 0 else 4
                yield StreamingInput(prompt=prompt, sampling_params=turn_params)
                tokens, finish = await replies.get()
                replay_token = tokens[-1] if finish == "length" else None
            producer_finished.set()

        text = ""
        tokens: list[int] = []
        async with asyncio.timeout(180):
            async for output in engine.generate(inputs(), params, "bounded-text"):
                turn = len(completed)
                assert turn < turns
                assert not output.finished
                assert output.prompt_token_ids == prompts[turn]
                assert len(output.outputs) == 1
                completion = output.outputs[0]
                assert completion.logprobs is not None
                assert len(completion.logprobs) == len(completion.token_ids)
                text += completion.text
                tokens.extend(completion.token_ids)
                if completion.finish_reason is None:
                    continue
                expected = "A" if turn % 2 == 0 else "B"
                assert text == expected, (
                    f"turn={turn} expected={expected!r} actual={text!r} "
                    f"tokens={tokens} finish={completion.finish_reason!r}"
                )
                assert 0 < len(tokens) <= (1 if turn % 2 == 0 else 4)
                assert completion.finish_reason in ("stop", "length")
                finishes.add(completion.finish_reason)
                completed.append(text)
                replies.put_nowait((tokens, completion.finish_reason))
                text, tokens = "", []
        assert producer_finished.is_set()
        assert len(completed) == turns
        assert finishes == {"stop", "length"}
        assert sum(map(len, prompts)) > 2 * retention.max_session_tokens
        assert engine.get_num_unfinished_requests() == 0
    finally:
        engine.shutdown()
    captured = capfd.readouterr()
    surviving = re.findall(
        r"Consolidating streaming session .*surviving_tokens=(\d+)",
        captured.out + captured.err,
    )
    assert surviving, "The session must cross a real consolidation boundary"
    assert max(map(int, surviving)) <= retention.max_session_tokens
