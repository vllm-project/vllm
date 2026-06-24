# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    return False


def make_mock_tokenizer(vocab: dict[str, int]) -> MagicMock:
    """Create a mock tokenizer with the given special-token vocabulary.

    The returned mock supports get_vocab(), encode(), and decode().
    decode() maps known token IDs back to their text and falls back to
    chr(id) for ASCII IDs or ``<id>`` for others.
    """
    id_to_text = {v: k for k, v in vocab.items()}
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = dict(vocab)
    tokenizer.decode.side_effect = lambda ids: "".join(
        id_to_text.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
    )
    return tokenizer


@pytest.fixture
def mock_request():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    return req


# ── Shared test configs ──────────────────────────────────────────────

VOCAB: dict[str, int] = {
    "<think>": 200,
    "</think>": 201,
    "<tool_call>": 202,
    "</tool_call>": 203,
}


def combined_config() -> ParserEngineConfig:
    """Config with reasoning tags and tool-call tags."""
    return ParserEngineConfig(
        name="combined_test",
        terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        token_id_terminals={
            "THINK_START": "<think>",
            "THINK_END": "</think>",
            "TOOL_START": "<tool_call>",
            "TOOL_END": "</tool_call>",
        },
        transitions={
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
        },
        initial_state=ParserState.REASONING,
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
    )


class CombinedTestEngine(ParserEngine):
    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(
            tokenizer, tools, parser_engine_config=combined_config(), **kwargs
        )


CombinedReasoningAdapter, CombinedToolAdapter = make_adapters(CombinedTestEngine)


class CombinedDelegating(DelegatingParser):
    reasoning_parser_cls = CombinedReasoningAdapter
    tool_parser_cls = CombinedToolAdapter
