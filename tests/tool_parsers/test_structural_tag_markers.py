# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The tool-call grammar must only admit encodings the parser recognises.

xgrammar's builtin structural tags spell markers such as ``<tool_call>`` as
strings, so the grammar also accepts the marker assembled from ordinary
sub-tokens. The streaming parser engine keys those markers by dedicated token
ID and treats the assembled spelling as content. ``bind_marker_tokens`` closes
that gap by matching the markers at the token level.
"""

import pytest
import xgrammar as xgr
from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
    TokenTriggeredTagsFormat,
    TriggeredTagsFormat,
)

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.structural_tag_markers import bind_marker_tokens

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
VOCAB = {"<tool_call>": 1, "</tool_call>": 2, "<think>": 3, "</think>": 4}
MARKERS = frozenset(VOCAB)


def _tag(begin: str, end: str) -> TagFormat:
    return TagFormat(begin=begin, content=JSONSchemaFormat(json_schema=SCHEMA), end=end)


class TestBindMarkerTokens:
    def test_triggered_tags_dispatch_on_marker_token(self):
        tag = StructuralTag(
            format=TriggeredTagsFormat(
                triggers=["<tool_call>"],
                tags=[_tag("<tool_call>get_weather", "</tool_call>")],
                excludes=["<think>", "</think>", "</tool_call>", "<arg_key>"],
                at_least_one=True,
            )
        )

        bound = bind_marker_tokens(tag, MARKERS, VOCAB).format

        assert isinstance(bound, TokenTriggeredTagsFormat)
        assert bound.trigger_tokens == ["<tool_call>"]
        assert bound.at_least_one is True
        # "<arg_key>" is not a token in this vocabulary, so it cannot be
        # excluded at the token level.
        assert bound.exclude_tokens == ["<think>", "</think>", "</tool_call>"]
        (inner,) = bound.tags
        assert inner.begin == TokenFormat(token="<tool_call>")
        assert inner.end == TokenFormat(token="</tool_call>")
        assert inner.content == SequenceFormat(
            elements=[
                ConstStringFormat(value="get_weather"),
                JSONSchemaFormat(json_schema=SCHEMA),
            ]
        )

    def test_trigger_longer_than_marker_keeps_remaining_text(self):
        """Qwen triggers on ``<tool_call>\\n<function=``; the tail stays literal."""
        tag = StructuralTag(
            format=TriggeredTagsFormat(
                triggers=["<tool_call>\n<function="],
                tags=[
                    _tag(
                        "<tool_call>\n<function=get_weather>\n",
                        "\n</function>\n</tool_call>",
                    )
                ],
            )
        )

        bound = bind_marker_tokens(tag, MARKERS, VOCAB).format

        assert isinstance(bound, TokenTriggeredTagsFormat)
        (inner,) = bound.tags
        assert inner.begin == TokenFormat(token="<tool_call>")
        assert inner.end == TokenFormat(token="</tool_call>")
        assert inner.content == SequenceFormat(
            elements=[
                ConstStringFormat(value="\n<function=get_weather>\n"),
                JSONSchemaFormat(json_schema=SCHEMA),
                ConstStringFormat(value="\n</function>\n"),
            ]
        )

    def test_literals_and_nested_separator_tags(self):
        tag = StructuralTag(
            format=SequenceFormat(
                elements=[
                    ConstStringFormat(value="\n\n<think>\n"),
                    TagsWithSeparatorFormat(
                        tags=[_tag("<tool_call>get_weather", "</tool_call>")],
                        separator="",
                        at_least_one=True,
                    ),
                    ConstStringFormat(value="</think>"),
                ]
            )
        )

        bound = bind_marker_tokens(tag, MARKERS, VOCAB).format

        assert isinstance(bound, SequenceFormat)
        head, calls, tail = bound.elements
        assert head == SequenceFormat(
            elements=[
                ConstStringFormat(value="\n\n"),
                TokenFormat(token="<think>"),
                ConstStringFormat(value="\n"),
            ]
        )
        assert tail == TokenFormat(token="</think>")
        assert isinstance(calls, TagsWithSeparatorFormat)
        assert calls.tags[0].begin == TokenFormat(token="<tool_call>")
        assert calls.tags[0].end == TokenFormat(token="</tool_call>")

    def test_plain_text_trigger_keeps_string_dispatch(self):
        """Llama triggers on a JSON prefix, which is not a marker."""
        fmt = TriggeredTagsFormat(
            triggers=['{"name": '],
            tags=[_tag('{"name": "get_weather", "parameters": ', "}")],
        )

        bound = bind_marker_tokens(StructuralTag(format=fmt), MARKERS, VOCAB)

        assert bound.format == fmt

    def test_markers_missing_from_vocab_leave_tag_untouched(self):
        tag = StructuralTag(
            format=TriggeredTagsFormat(
                triggers=["<tool_call>"],
                tags=[_tag("<tool_call>get_weather", "</tool_call>")],
            )
        )

        assert bind_marker_tokens(tag, MARKERS, {"<think>": 3}) is tag
        assert bind_marker_tokens(tag, frozenset(), VOCAB) is tag

    def test_end_alternatives_stay_strings(self):
        fmt = TagFormat(
            begin="<tool_call>",
            content=AnyTextFormat(),
            end=["</tool_call>", "<|eot|>"],
        )

        bound = bind_marker_tokens(StructuralTag(format=fmt), MARKERS, VOCAB)

        assert isinstance(bound.format, TagFormat)
        assert bound.format.begin == TokenFormat(token="<tool_call>")
        assert bound.format.end == ["</tool_call>", "<|eot|>"]

    def test_idempotent(self):
        tag = StructuralTag(
            format=TriggeredTagsFormat(
                triggers=["<tool_call>"],
                tags=[_tag("<tool_call>get_weather", "</tool_call>")],
                excludes=["<think>"],
            )
        )

        once = bind_marker_tokens(tag, MARKERS, VOCAB)

        assert bind_marker_tokens(once, MARKERS, VOCAB) == once


def test_parser_engine_reports_markers_present_in_vocab():
    tokenizer = make_mock_tokenizer({"<tool_call>": 1, "</tool_call>": 2})
    parser = Glm47MoeParser(tokenizer, chat_template_kwargs={"thinking": False})

    # <think>/</think> are token-ID terminals too, but this vocabulary has no
    # dedicated token for them, so the engine matches them as text.
    assert parser.token_id_markers == {"<tool_call>", "</tool_call>"}


# --------------------------------------------------------------------------
# End to end: grammar and parser agree on real tokenizers.
# --------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": SCHEMA}},
    {"type": "function", "function": {"name": "get_time", "parameters": SCHEMA}},
]

# (tool parser, tokenizer, tool-call body without the <tool_call> markers)
CASES = {
    "qwen3": (
        Qwen3EngineToolParser,
        "Qwen/Qwen3-0.6B",
        "\n<function=get_weather>\n<parameter=city>\nSydney\n</parameter>\n"
        "</function>\n",
    ),
    "glm47": (
        Glm47MoeModelToolParser,
        "zai-org/GLM-4.7-Flash",
        "get_weather<arg_key>city</arg_key><arg_value>Sydney</arg_value>",
    ),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    parser_cls, model, body = CASES[request.param]
    tokenizer = get_tokenizer(model)
    return parser_cls(tokenizer), tokenizer, body


def _encodings(tokenizer, body: str) -> tuple[list[int], list[int]]:
    """The same tool call with dedicated marker tokens and with split ones."""

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    start = tokenizer.get_vocab()["<tool_call>"]
    end = tokenizer.get_vocab()["</tool_call>"]
    atomic = [start, *enc(body), end]
    split = [*enc("<"), *enc("tool_call"), *enc(">"), *enc(body)]
    split += [*enc("</"), *enc("tool_call"), *enc(">")]
    assert tokenizer.decode(atomic) == tokenizer.decode(split)
    assert start not in split and end not in split
    return atomic, split


def _accepts(tokenizer, tag: StructuralTag, token_ids: list[int]) -> bool:
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))
    matcher = xgr.GrammarMatcher(compiler.compile_structural_tag(tag))
    return all(matcher.accept_token(token_id) for token_id in token_ids)


def _parse(parser, tokenizer, token_ids: list[int]) -> tuple[list[str], str]:
    """Stream ``token_ids`` through the parser: (tool names, content)."""
    request = ChatCompletionRequest(messages=[], model="m", tools=TOOLS)
    previous_text = ""
    previous_ids: list[int] = []
    deltas = []
    for token_id in token_ids:
        current_ids = previous_ids + [token_id]
        current_text = tokenizer.decode(current_ids)
        deltas.append(
            parser.extract_tool_calls_streaming(
                previous_text,
                current_text,
                current_text[len(previous_text) :],
                previous_ids,
                current_ids,
                [token_id],
                request,
            )
        )
        previous_text, previous_ids = current_text, current_ids
    deltas.append(parser.finish_streaming())
    deltas = [delta for delta in deltas if delta is not None]
    names = [
        tool_call.function.name
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function and tool_call.function.name
    ]
    content = "".join(delta.content or "" for delta in deltas)
    return names, content


@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get_weather"}}],
    ids=["required", "named"],
)
def test_constrained_grammar_only_admits_parseable_markers(case, tool_choice):
    parser, tokenizer, body = case
    atomic, split = _encodings(tokenizer, body)
    request = ChatCompletionRequest(
        messages=[], model="m", tools=TOOLS, tool_choice=tool_choice
    )

    tag = parser.get_structural_tag(request)

    assert tag is not None
    # What the grammar admits, the parser parses cleanly ...
    assert _accepts(tokenizer, tag, atomic)
    assert _parse(parser, tokenizer, atomic) == (["get_weather"], "")
    # ... and the encoding the parser cannot parse cleanly (GLM finds no
    # tool call at all, Qwen leaks the marker into content) is rejected.
    assert _parse(parser, tokenizer, split) != (["get_weather"], "")
    assert not _accepts(tokenizer, tag, split)


def test_auto_grammar_keeps_free_text_and_dedicated_calls(case):
    parser, tokenizer, body = case
    atomic, _ = _encodings(tokenizer, body)
    strict_tools = [
        {"type": "function", "function": {**tool["function"], "strict": True}}
        for tool in TOOLS
    ]
    request = ChatCompletionRequest(
        messages=[], model="m", tools=strict_tools, tool_choice="auto"
    )

    tag = parser.get_structural_tag(request)

    assert tag is not None
    prose = tokenizer.encode("Sure, one moment.", add_special_tokens=False)
    assert _accepts(tokenizer, tag, prose)
    assert _accepts(tokenizer, tag, [*prose, *atomic, *prose])
    # The string excludes of the builtin tag became token excludes. Where the
    # builtin forbids </tool_call> in free text, the dedicated token is still
    # rejected ...
    end = tokenizer.get_vocab()["</tool_call>"]
    assert isinstance(tag.format, TokenTriggeredTagsFormat)
    if "</tool_call>" in tag.format.exclude_tokens:
        assert not _accepts(tokenizer, tag, [*prose, end])
    # ... while the same text spelled from ordinary tokens is admitted, and
    # the parser reads it as content, which is what the token excludes model.
    split_end = [
        *tokenizer.encode("</", add_special_tokens=False),
        *tokenizer.encode("tool_call", add_special_tokens=False),
        *tokenizer.encode(">", add_special_tokens=False),
    ]
    assert end not in split_end
    stray = [*prose, *split_end, *prose]
    assert _accepts(tokenizer, tag, stray)
    assert _parse(parser, tokenizer, stray) == ([], tokenizer.decode(stray))
