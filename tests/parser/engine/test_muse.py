# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse parser (Muse Glimmer and Muse Spark Open): recipient-routed messages.

Covers routing in both parse paths, and where the structured output grammar
starts: at the `to=user` answer header, not at a reasoning, commentary or tool
header, so a turn that ends in a tool call is not forced into the schema.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.replay_harness import (
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.entrypoints.generate.base.protocol import parser_chat_template_kwargs
from vllm.parser.engine.registered_adapters import (
    MuseParserToolAdapter,
)
from vllm.parser.muse import MuseParser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.muse_engine_reasoning_parser import MuseEngineReasoningParser
from vllm.tool_parsers import ToolParserManager

VOCAB = {
    "<|start|>": 200022,
    "<|message|>": 200023,
    "<|eom|>": 200007,
    "<|eot|>": 200008,
}
JSON_FORMAT = {"type": "json_schema", "json_schema": {"name": "a", "schema": {}}}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
]


def _msg(recipient: str, body: str, end: str = "<|eom|>") -> str:
    return f" to={recipient}<|message|>{body}{end}"


def _turn(*messages: str) -> str:
    return "<|start|>assistant".join(messages)


def _call(city: str) -> str:
    return _msg(
        "get_weather",
        '<atem:function_calls><atem:invoke name="get_weather">'
        f'<atem:parameter name="city">{city}</atem:parameter>'
        "</atem:invoke></atem:function_calls>",
    )


# (model output, reasoning, content, tool-call cities)
CASES = {
    "answer": (_msg("user", "391.", "<|eot|>"), "", "391.", []),
    "reasoning_answer": (
        _turn(_msg("self", "17*23."), _msg("user", "391.", "<|eot|>")),
        "17*23.",
        "391.",
        [],
    ),
    "commentary_answer": (
        _turn(
            _msg("self", "Plan."),
            _msg("commentary", "Working. "),
            _msg("user", "391.", "<|eot|>"),
        ),
        "Plan.",
        "Working. 391.",
        [],
    ),
    "reasoning_tool": (
        _turn(_msg("self", "Need the tool."), _call("Paris")),
        "Need the tool.",
        "",
        ["Paris"],
    ),
    # Muse Glimmer may write the call inside its answer.
    "tool_in_answer": (
        _turn(
            _msg("self", "Need the tool."),
            _msg("user", "Checking." + _call("Paris").split("<|message|>")[1]),
        ),
        "Need the tool.",
        "Checking.",
        ["Paris"],
    ),
    "tool_text_in_reasoning": (
        _turn(
            _msg("self", "Maybe <atem:function_calls> later."),
            _msg("user", "No call.", "<|eot|>"),
        ),
        "Maybe <atem:function_calls> later.",
        "No call.",
        [],
    ),
    "tool_then_more_reasoning": (
        _turn(
            _msg("self", "Plan."),
            _call("Paris"),
            _msg("self", "Again."),
            _call("Rome"),
        ),
        "Plan.Again.",
        "",
        ["Paris", "Rome"],
    ),
    "answer_then_tool": (
        _turn(_msg("user", "A"), _call("Paris")),
        "",
        "A",
        ["Paris"],
    ),
    "truncated_header": (
        _msg("self", "Plan.") + "<|start|>assistant to=us",
        "Plan.",
        "",
        [],
    ),
}


def _tokens(text: str) -> list[tuple[int, str]]:
    tokens, i = [], 0
    while i < len(text):
        marker = next((m for m in VOCAB if text.startswith(m, i)), None)
        if marker is not None:
            tokens.append((VOCAB[marker], marker))
            i += len(marker)
        else:
            tokens.append((ord(text[i]), text[i]))
            i += 1
    return tokens


def _ids(text: str) -> list[int]:
    return [token_id for token_id, _ in _tokens(text)]


PROMPT = _ids("<|start|>user<|message|>hi<|eot|><|start|>assistant")


@pytest.fixture
def tokenizer():
    return make_mock_tokenizer(VOCAB)


@pytest.fixture
def parser_cls():
    return ParserManager.get_parser(
        tool_parser_name="muse", reasoning_parser_name="muse", enable_auto_tools=True
    )


def test_parser_names_resolve_to_engine_adapters():
    for name in ("muse", "musespark"):
        assert issubclass(
            ToolParserManager.get_tool_parser(name), MuseParserToolAdapter
        )
        assert (
            ReasoningParserManager.get_reasoning_parser(name)
            is MuseEngineReasoningParser
        )


@pytest.mark.parametrize("case", CASES)
def test_parse(parser_cls, tokenizer, case):
    text, reasoning, content, cities = CASES[case]
    parsed_reasoning, parsed_content, tool_calls = parser_cls(tokenizer).parse(
        text, _test_request(TOOLS), enable_auto_tools=True
    )
    assert (parsed_reasoning or "") == reasoning
    assert (parsed_content or "") == content
    assert [json.loads(tc.arguments)["city"] for tc in tool_calls or []] == cities


@pytest.mark.parametrize("chunk_size", range(1, 24))
@pytest.mark.parametrize("case", CASES)
def test_streaming_matches_parse(parser_cls, tokenizer, case, chunk_size):
    text, reasoning, content, cities = CASES[case]
    output = collect_output(
        replay_streaming(
            parser_cls(tokenizer),
            _tokens(text),
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=TOOLS,
        )
    )
    assert output.reasoning == reasoning
    assert output.content == content
    assert [json.loads(tc["arguments"])["city"] for tc in output.tool_calls] == cities


def test_header_pending_when_reasoning_ends(parser_cls, tokenizer):
    # The first batch ends reasoning at the tool header and stops inside the
    # next message's header; the tool pass must still route that header.
    text = CASES["tool_then_more_reasoning"][0]
    tokens = _tokens(text)
    split = [i for i, (t, _) in enumerate(tokens) if t == VOCAB["<|message|>"]][2]
    output = collect_output(
        replay_streaming(
            parser_cls(tokenizer),
            tokens,
            chunk_size=split,
            finished_on_last=True,
            tools=TOOLS,
        )
    )
    assert (output.reasoning, output.content) == ("Plan.Again.", "")
    assert [json.loads(tc["arguments"])["city"] for tc in output.tool_calls] == [
        "Paris",
        "Rome",
    ]


@pytest.mark.parametrize("chunk_size", range(1, 12))
def test_streaming_without_tool_parser(tokenizer, chunk_size):
    # Without --enable-auto-tool-choice the reasoning parser alone parses
    # every message, including those after the answer starts.
    parser_cls = ParserManager.get_parser(reasoning_parser_name="muse")
    assert parser_cls is not None
    text = _turn(_msg("self", "Plan."), _msg("user", "A"), _msg("user", "B", "<|eot|>"))
    output = collect_output(
        replay_streaming(
            parser_cls(tokenizer),
            _tokens(text),
            chunk_size=chunk_size,
            finished_on_last=True,
            prompt_token_ids=PROMPT,
        )
    )
    assert (output.reasoning, output.content) == ("Plan.", "AB")


@pytest.mark.parametrize("chunk_size", [None, 1, 3])
def test_commentary_is_reasoning_under_json_response_format(tokenizer, chunk_size):
    # With structured output, `content` is the constrained answer alone.
    text = CASES["commentary_answer"][0]
    kwargs = {"chat_template_kwargs": {"response_format": JSON_FORMAT}}
    if chunk_size is None:
        reasoning, content = MuseParser(tokenizer, **kwargs).extract_reasoning(
            text, _test_request()
        )
    else:
        output = collect_output(
            replay_streaming(
                MuseParser(tokenizer, **kwargs),
                _tokens(text),
                chunk_size=chunk_size,
                finished_on_last=True,
            )
        )
        reasoning, content = output.reasoning, output.content
    assert (reasoning, content) == ("Plan.Working. ", "391.")


@pytest.mark.parametrize("client_value", ["json", {}, {"type": "json_object"}])
def test_client_response_format_kwarg_does_not_reach_the_parser(
    tokenizer, client_value
):
    # Clients control `chat_template_kwargs`; only the validated request field
    # may select commentary routing.
    kwargs = parser_chat_template_kwargs(None, {"response_format": client_value})
    reasoning, content = MuseParser(
        tokenizer, chat_template_kwargs=kwargs
    ).extract_reasoning(CASES["commentary_answer"][0], _test_request())
    assert (reasoning, content) == ("Plan.", "Working. 391.")


def test_grammar_starts_at_the_answer_header(tokenizer):
    parser = MuseParser(tokenizer)
    reasoning = _ids(_msg("self", "Plan."))
    commentary = _ids("<|start|>assistant to=commentary<|message|>")
    answer = _ids("<|start|>assistant to=user<|message|>")
    tool = _ids("<|start|>assistant to=get_weather<|message|>")

    def ends(*parts: list[int]) -> bool:
        ids = [t for part in parts for t in part]
        return parser.is_reasoning_end_streaming(ids, parts[-1])

    assert not parser.is_reasoning_end(PROMPT)
    assert not ends(PROMPT, reasoning)
    assert not ends(PROMPT, reasoning, commentary)
    assert ends(PROMPT, reasoning, answer)
    assert not ends(PROMPT, reasoning, tool)
    # A speculative suffix that opens the answer and then more text still ends
    # reasoning at the answer header.
    assert ends(PROMPT, reasoning, answer + _ids('{"a": 1}'))
    # A prompt that already ends with the answer header (a prefilled answer).
    assert parser.is_reasoning_end(PROMPT + reasoning + answer)
