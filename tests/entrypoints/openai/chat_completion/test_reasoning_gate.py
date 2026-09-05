# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The structured-output reasoning gate (``reasoning_ended`` passed to
``engine.generate``) must be decided by the reasoning parser when one is
configured — including engine-based parsers, which expose no
``reasoning_parser`` attribute — even when the client hides reasoning
(``include_reasoning=False``). Otherwise a grammar is enforced from the first
token while an always-thinking model (GLM-5.3) is still inside ``<think>``."""

from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.entrypoints.openai.chat_completion.test_serving_chat import (
    MODEL_NAME,
    MockModelConfig,
    _build_renderer,
    _build_serving_chat,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM

JSON_SCHEMA_RF = {
    "type": "json_schema",
    "json_schema": {
        "name": "weather",
        "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


class _EngineStyleParser:
    """Mimics a parser-engine class returned by ParserManager.get_parser when
    tool and reasoning parsing share one engine: no ``reasoning_parser``
    attribute, but ``is_reasoning_end`` implemented on the object itself."""

    reasoning_parser = None

    def __init__(self, tokenizer, tools=None, **kwargs):
        pass

    def is_reasoning_end(self, input_ids):
        return False  # prompt ends with <think>: reasoning has not ended

    def adjust_request(self, request):
        return request


def _engine():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)

    async def mock_generate(*args, **kwargs):
        yield RequestOutput(
            request_id="test-request",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text='{"city": "London"}',
                    token_ids=[4, 5, 6],
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            finished=True,
        )

    mock_engine.generate = AsyncMock(side_effect=mock_generate)
    return mock_engine


async def _run(serving_chat, **request_fields):
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Weather in London?"}],
        response_format=JSON_SCHEMA_RF,
        **request_fields,
    )
    raw = MagicMock()
    raw.headers = {}
    raw.state = MagicMock()
    with suppress(Exception):
        await serving_chat.create_chat_completion(req, raw)


@pytest.mark.asyncio
async def test_hidden_reasoning_defers_to_engine_parser():
    engine = _engine()
    serving_chat = _build_serving_chat(engine)
    serving_chat.parser_cls = _EngineStyleParser
    serving_chat._has_reasoning_parser = True

    await _run(serving_chat, include_reasoning=False)
    assert engine.generate.call_args.kwargs["reasoning_ended"] is False


@pytest.mark.asyncio
async def test_visible_reasoning_defers_to_engine_parser():
    engine = _engine()
    serving_chat = _build_serving_chat(engine)
    serving_chat.parser_cls = _EngineStyleParser
    serving_chat._has_reasoning_parser = True

    await _run(serving_chat, include_reasoning=True)
    assert engine.generate.call_args.kwargs["reasoning_ended"] is False


@pytest.mark.asyncio
async def test_hidden_reasoning_without_reasoning_parser_keeps_shortcut():
    engine = _engine()
    serving_chat = _build_serving_chat(engine)  # no reasoning parser configured

    await _run(serving_chat, include_reasoning=False)
    assert engine.generate.call_args.kwargs["reasoning_ended"] is True
