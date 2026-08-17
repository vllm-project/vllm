# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Gemma4 streaming end-of-turn handling."""

from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.parser.parser_manager import ParserManager

CHANNEL_START_ID = 100
CHANNEL_END_ID = 101
TOOL_CALL_START_ID = 48
TOOL_CALL_END_ID = 50
TOOL_RESPONSE_START_ID = 102
TOOL_RESPONSE_END_ID = 103
TURN_END_ID = 106
REGULAR_ID = 200

SPECIAL_TOKEN_MAP = {
    CHANNEL_START_ID: "<|channel>",
    CHANNEL_END_ID: "<channel|>",
    TOOL_CALL_START_ID: "<|tool_call>",
    TOOL_CALL_END_ID: "<tool_call|>",
    TOOL_RESPONSE_START_ID: "<|tool_response>",
    TOOL_RESPONSE_END_ID: "<tool_response|>",
    TURN_END_ID: "<turn|>",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "No-op tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        text: token_id for token_id, text in SPECIAL_TOKEN_MAP.items()
    }
    tokenizer.all_special_tokens = list(SPECIAL_TOKEN_MAP.values())
    tokenizer.all_special_ids = list(SPECIAL_TOKEN_MAP)

    decode_map = {
        **SPECIAL_TOKEN_MAP,
        REGULAR_ID: ".",
    }

    def decode(token_ids, skip_special_tokens=False):
        return "".join(
            ""
            if skip_special_tokens and token_id in SPECIAL_TOKEN_MAP
            else decode_map[token_id]
            for token_id in token_ids
        )

    tokenizer.decode.side_effect = decode
    return tokenizer


def _run_stream(enable_thinking: bool | None):
    tokenizer = _make_tokenizer()
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="gemma4",
        tool_parser_name="gemma4",
        enable_auto_tools=True,
        model_name="gemma4",
    )
    assert parser_cls is not None

    request = ChatCompletionRequest(
        model="gemma4",
        messages=[{"role": "user", "content": "replay"}],
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
    )

    chat_template_kwargs = (
        {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    )
    parser = parser_cls(
        tokenizer,
        tools=request.tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    request = parser.adjust_request(request)

    prompt_token_ids = [
        TOOL_RESPONSE_START_ID,
        REGULAR_ID,
        TOOL_RESPONSE_END_ID,
    ]
    first_delta = parser.parse_delta(
        delta_text=".",
        delta_token_ids=[REGULAR_ID],
        request=request,
        prompt_token_ids=prompt_token_ids,
        finished=False,
    )
    last_delta = parser.parse_delta(
        delta_text="",
        delta_token_ids=[TURN_END_ID],
        request=request,
        prompt_token_ids=None,
        finished=True,
    )
    return parser, first_delta, last_delta


def test_template_default_routes_turn_end_to_tool_parser():
    parser, first_delta, last_delta = _run_stream(enable_thinking=None)

    assert parser.tool_parser is not None
    assert parser._stream_state.reasoning_ended
    assert first_delta is not None
    assert first_delta.content == "."
    assert last_delta is None
