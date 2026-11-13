# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Muse Spark parser, streaming and non-streaming."""

import json

import pytest

from tests.parser.engine.replay_harness import (
    MockTokenizer,
    collect_output,
    replay_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager

EOM, EOT, START, MESSAGE = "<|eom|>", "<|eot|>", "<|start|>", "<|message|>"
VOCAB = {EOM: 200007, EOT: 200008, START: 200022, MESSAGE: 200023}
H = f"{START}assistant"


def _tool(name: str, properties: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


def _call(name: str, emitted: str | None = None, **params: str) -> str:
    body = "".join(
        f'<atem:parameter name="{key}">{value}</atem:parameter>\n'
        for key, value in params.items()
    )
    return (
        f" to={emitted or name}{MESSAGE}<atem:function_calls>\n"
        f'<atem:invoke name="{emitted or name}">\n{body}</atem:invoke>\n'
        "</atem:function_calls>"
    )


def _tokens(text: str) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    i = 0
    while i < len(text):
        marker = next((m for m in VOCAB if text.startswith(m, i)), None)
        if marker is not None:
            tokens.append((VOCAB[marker], marker))
            i += len(marker)
        else:
            tokens.append((ord(text[i]), text[i]))
            i += 1
    return tokens


def _muse_parser(tokens: list[tuple[int, str]], tools: list[dict] | None = None):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="muse", reasoning_parser_name="muse", enable_auto_tools=True
    )
    assert parser_cls is not None
    tool_params = [ChatCompletionToolsParam.model_validate(t) for t in tools or []]
    return parser_cls(MockTokenizer(dict(VOCAB), tokens), tool_params or None)


def _parse(
    raw: str, chunk: int | None, tools: list[dict] | None = None
) -> tuple[str, str, list[tuple[str, dict]]]:
    """Return reasoning, content and calls; `chunk=None` parses non-streaming."""
    tokens = _tokens(raw)
    parser = _muse_parser(tokens, tools)
    if chunk is None:
        request = ChatCompletionRequest(
            model="muse", messages=[{"role": "user", "content": "hi"}], tools=tools
        )
        reasoning, content, calls = parser.parse(
            raw,
            request,
            enable_auto_tools=True,
            model_output_token_ids=[tid for tid, _ in tokens],
        )
        found = [(c.name, json.loads(c.arguments)) for c in calls or []]
        return reasoning or "", content or "", found
    output = collect_output(
        replay_streaming(
            parser, tokens, chunk_size=chunk, finished_on_last=True, tools=tools
        )
    )
    found = [(c["name"], json.loads(c["arguments"])) for c in output.tool_calls]
    return output.reasoning, output.content, found


MODES = pytest.mark.parametrize("chunk", [1, 3, None], ids=["s1", "s3", "full"])


@MODES
@pytest.mark.parametrize(
    ("raw", "reasoning", "content"),
    [
        (
            f" to=self{MESSAGE}Plan.{EOM}{H} to=user{MESSAGE}Done.{EOT}",
            "Plan.",
            "Done.",
        ),
        (f" to=developer{MESSAGE}internal{EOM}{H} to=user{MESSAGE}Ok.", "", "Ok."),
        (f" to=commentary{MESSAGE}Wait. {EOM}{H} to=user{MESSAGE}Ok.", "", "Wait. Ok."),
        (f" to=self{MESSAGE}Think. to=user{MESSAGE}Done.{EOT}", "Think. ", "Done."),
        ("Plain text.", "", "Plain text."),
    ],
    ids=["self-user", "developer", "commentary", "missing-eom", "unframed"],
)
def test_recipient_routes_body(chunk, raw, reasoning, content):
    assert _parse(raw, chunk) == (reasoning, content, [])


@MODES
def test_tool_header_after_body_without_eom_keeps_call(chunk):
    raw = f" to=self{MESSAGE}Need the time." + _call("time_now") + EOT
    assert _parse(raw, chunk, [_tool("time_now")]) == (
        "Need the time. ",
        "",
        [("time_now", {})],
    )


@MODES
def test_atem_quoted_in_answer_is_not_a_call(chunk):
    quoted = _call("get_weather", city="Rome").split(MESSAGE, 1)[1]
    raw = f" to=user{MESSAGE}Write {quoted} to call it.{EOT}"
    assert _parse(raw, chunk, [_tool("get_weather")])[2] == []


@MODES
def test_arguments_follow_the_registered_tool(chunk):
    tool = _tool("echo", {"text": {"type": "string"}, "count": {"type": "integer"}})
    raw = _call("echo", "echo.echo", text="25", count="25") + EOT
    assert _parse(raw, chunk, [tool])[2] == [("echo", {"text": "25", "count": 25})]


def test_reasoning_ends_only_at_a_tool_header_of_the_current_turn():
    prompt = _tokens(
        f"{H}{_call('get_weather', city='Paris')}{EOT}"
        f"{START}tool get_weather{MESSAGE}21{EOT}{H}"
    )
    output = _tokens(f" to=self{MESSAGE}Check.{EOM}{H}" + _call("time_now"))
    reasoner_cls = ReasoningParserManager.get_reasoning_parser("muse")
    reasoner = reasoner_cls(MockTokenizer(dict(VOCAB), prompt + output))
    ids = [tid for tid, _ in prompt + output]
    assert not reasoner.is_reasoning_end(ids[: len(prompt)])
    ended = [
        reasoner.is_reasoning_end_streaming(ids[: end + 1], ids[end : end + 1])
        for end in range(len(prompt), len(ids))
    ]
    tool_message = len(ids) - ids[::-1].index(VOCAB[MESSAGE]) - 1
    assert [i for i, e in enumerate(ended) if e] == [tool_message - len(prompt)]


def test_reasoning_tokens_include_reasoning_after_tool_call():
    tokens = _tokens(
        f" to=self{MESSAGE}ab{EOM}{H}"
        + _call("time_now")
        + f"{EOM}{H} to=self{MESSAGE}cde{EOM}{H} to=user{MESSAGE}Ok.{EOT}"
    )
    parser = _muse_parser(tokens)
    replay_streaming(parser, tokens, chunk_size=1, finished_on_last=True)
    assert parser.count_reasoning_tokens([tid for tid, _ in tokens]) == len("abcde")
