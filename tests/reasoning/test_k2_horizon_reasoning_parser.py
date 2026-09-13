# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import string
from collections.abc import Sequence

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.k2_horizon_reasoning_parser import K2HorizonReasoningParser

pytestmark = pytest.mark.skip_global_cleanup

EFFORT_TOKENS = {
    "high": ("<ifm|think>", "</ifm|think>"),
    "medium": ("<ifm|think_fast>", "</ifm|think_fast>"),
    "low": ("<ifm|think_faster>", "</ifm|think_faster>"),
}
TOOL_CALLS_START = "<ifm|tool_calls>"


class K2Tokenizer:
    def __init__(self) -> None:
        special_tokens = [
            token for token_pair in EFFORT_TOKENS.values() for token in token_pair
        ] + [TOOL_CALLS_START]
        tokens = special_tokens + list(string.printable)
        self._vocab = {token: index for index, token in enumerate(tokens, start=1)}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        position = 0
        specials = sorted(EFFORT_TOKENS["high"] + (TOOL_CALLS_START,), key=len)
        while position < len(text):
            for special in specials:
                if text.startswith(special, position):
                    tokens.append(self._vocab[special])
                    position += len(special)
                    break
            else:
                tokens.append(self._vocab[text[position]])
                position += 1
        return tokens

    def decode(
        self,
        ids: Sequence[int] | int,
        skip_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        inverse_vocab = {token_id: token for token, token_id in self._vocab.items()}
        return "".join(inverse_vocab[token_id] for token_id in ids)


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="k2-horizon",
        messages=[{"role": "user", "content": "test"}],
        **kwargs,
    )


@pytest.mark.parametrize("effort", ["high", "medium", "low"])
def test_reasoning_effort_selects_tokens(effort: str):
    parser = K2HorizonReasoningParser(
        K2Tokenizer(),
        chat_template_kwargs={"reasoning_effort": effort},
    )

    assert (parser.start_token, parser.end_token) == EFFORT_TOKENS[effort]


def test_reasoning_effort_precedence_and_default():
    tokenizer = K2Tokenizer()
    request = _request(
        reasoning_effort="medium",
        chat_template_kwargs={"reasoning_effort": "low"},
    )
    resolved_kwargs = (
        request.build_chat_params(None, "auto")
        .with_defaults({"reasoning_effort": "high"})
        .chat_template_kwargs
    )

    parser = K2HorizonReasoningParser(
        tokenizer,
        chat_template_kwargs=resolved_kwargs,
    )
    default_parser = K2HorizonReasoningParser(tokenizer)

    assert parser.start_token == EFFORT_TOKENS["medium"][0]
    assert default_parser.start_token == EFFORT_TOKENS["high"][0]


@pytest.mark.parametrize("effort", ["none", "unknown", None])
def test_invalid_reasoning_effort_raises(effort: str | None):
    with pytest.raises(ValueError, match="Unsupported reasoning_effort"):
        K2HorizonReasoningParser(
            K2Tokenizer(),
            chat_template_kwargs={"reasoning_effort": effort},
        )


def test_non_streaming_explicit_and_implicit_boundaries():
    parser = K2HorizonReasoningParser(K2Tokenizer())
    request = _request()

    assert parser.extract_reasoning("<ifm|think>plan</ifm|think>answer", request) == (
        "plan",
        "answer",
    )
    assert parser.extract_reasoning(
        f"plan{TOOL_CALLS_START}<ifm|tool_call>ping</ifm|tool_call>", request
    ) == (
        "plan",
        f"{TOOL_CALLS_START}<ifm|tool_call>ping</ifm|tool_call>",
    )


@pytest.mark.parametrize("reasoning", ["plan", " ", "\n", " \n"])
def test_character_split_reasoning_stream(reasoning: str):
    parser = K2HorizonReasoningParser(K2Tokenizer())
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    for char in f"{reasoning}</ifm|think>answer":
        delta = parser.extract_reasoning_streaming(
            previous_text="",
            current_text="",
            delta_text=char,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        if delta is not None:
            if delta.reasoning is not None:
                reasoning_parts.append(delta.reasoning)
            if delta.content is not None:
                content_parts.append(delta.content)

    assert "".join(reasoning_parts) == reasoning
    assert "".join(content_parts) == "answer"


def test_latest_boundary_controls_reasoning_state():
    tokenizer = K2Tokenizer()
    parser = K2HorizonReasoningParser(tokenizer)
    vocab = tokenizer.get_vocab()

    assert parser.is_reasoning_end([vocab["<ifm|think>"], vocab[TOOL_CALLS_START]])
    assert not parser.is_reasoning_end([vocab[TOOL_CALLS_START], vocab["<ifm|think>"]])


def test_composed_streaming_reasoning_to_tool_handoff():
    tokenizer = K2Tokenizer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    request = _request(tools=tools, tool_choice="auto")
    parser_cls = ParserManager.get_parser(
        tool_parser_name="k2_horizon",
        reasoning_parser_name="k2_horizon",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(
        tokenizer,
        request.tools,
        chat_template_kwargs={"reasoning_effort": "high"},
    )
    output = "plan<ifm|tool_calls><ifm|tool_call>ping</ifm|tool_call></ifm|tool_calls>"
    reasoning_parts: list[str] = []
    tool_names: list[str] = []

    for index, char in enumerate(output):
        delta = parser.parse_delta(
            delta_text=char,
            delta_token_ids=tokenizer.encode(char),
            request=request,
            prompt_token_ids=[] if index == 0 else None,
            finished=index == len(output) - 1,
        )
        if delta is not None:
            if delta.reasoning is not None:
                reasoning_parts.append(delta.reasoning)
            tool_names.extend(
                tool_call.function.name
                for tool_call in delta.tool_calls
                if tool_call.function is not None
                and tool_call.function.name is not None
            )

    assert "".join(reasoning_parts) == "plan"
    assert tool_names == ["ping"]


def test_reasoning_parser_registered():
    assert (
        ReasoningParserManager.get_reasoning_parser("k2_horizon")
        is K2HorizonReasoningParser
    )
