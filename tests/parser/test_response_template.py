# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types.responses import NamespaceTool

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.exceptions import VLLMValidationError
from vllm.parser import response_template as response_template_module
from vllm.parser.gemma4 import Gemma4Parser
from vllm.parser.parser_manager import ParserManager
from vllm.parser.response_template import (
    ResponseTemplateParser,
    ResponseTemplateReasoningParser,
    ResponseTemplateToolParser,
    resolve_response_template,
)
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

# Public metadata shared by the Gemma 4 instruction checkpoints.
GEMMA4_RESPONSE_TEMPLATE: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "fields": {
        "content": {
            "close": ["<turn|>", "<|tool_response>", "<eos>"],
            "content": "text",
        },
        "thinking": {
            "close": "<channel|>",
            "content": "text",
            "open": "<|channel>thought\n",
        },
        "tool_calls": {
            "close": "<tool_call|>",
            "content": "json",
            "content_args": {
                "string_delims": [['<|"|>', '<|"|>']],
                "unquoted_keys": True,
            },
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "repeats": True,
            "transform": {
                "function": {
                    "arguments": "{content}",
                    "name": "{name}",
                },
                "type": "function",
            },
        },
    },
    "start_anchor": ["<|turn>model\n", "<tool_response|>"],
}

PREFIX = "<|turn>model\n"
THINKING = "<|channel>thought\nplan<channel|>"
TOOL_GENERATION = (
    '<|tool_call>call:set_alarm{hour:7,label:<|"|>morning<|"|>}<tool_call|>'
)

TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "set_alarm",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {
                        "oneOf": [
                            {"type": "integer"},
                            {"type": "null"},
                        ]
                    },
                    "label": {"type": "string"},
                },
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                },
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "set_status",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_active": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "score": {"type": "number"},
                },
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "complex_function",
            "parameters": {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {"inner": {"type": "string"}},
                    },
                    "list": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "configure",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "ratio": {"type": "number"},
                    "label": {"type": "string"},
                    "value": {"type": ["string", "null"]},
                },
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_status",
            "parameters": {"type": "object", "properties": {}},
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get-weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    ),
    ChatCompletionToolsParam(
        type="function",
        function={
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "language": {"type": "string"},
                            "min_stars": {"type": "integer"},
                        },
                    },
                },
            },
        },
    ),
]

VOCAB = {
    "<|tool_call>": 48,
    "<tool_call|>": 49,
    "<|channel>": 50,
    "<channel|>": 51,
    '<|"|>': 52,
    "<|turn>": 53,
    "<|tool_response>": 54,
    "<turn|>": 55,
}
PROMPT_TOKEN_IDS = [1]
OPEN_REASONING_PROMPT_TOKEN_IDS = [53, 100, 50, 101]
PROMPT_TOKEN_TEXT = {
    100: "model\n",
    101: "thought\n",
    102: "plan",
    103: "answer",
}


class FakeTokenizer:
    def __init__(self, response_template=None, *, prefix=PREFIX):
        self.response_template = response_template
        self.init_kwargs = {}
        self.name_or_path = ""
        self.prefix = prefix
        self.all_special_tokens = list(VOCAB)
        self.all_special_ids = list(VOCAB.values())

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        if token_ids == [2]:
            return PREFIX + "answer"
        if token_ids == OPEN_REASONING_PROMPT_TOKEN_IDS:
            return PREFIX + "<|channel>thought\n"
        if token_ids != PROMPT_TOKEN_IDS:
            inverse_vocab = {token_id: token for token, token_id in VOCAB.items()}
            return "".join(
                PROMPT_TOKEN_TEXT.get(token_id, inverse_vocab.get(token_id, ""))
                for token_id in token_ids
            )
        return self.prefix if token_ids else ""

    def get_vocab(self):
        return VOCAB

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "plan":
            return [102]
        return list(text.encode())


def request(tool_choice="auto", *, include_reasoning=True, **overrides):
    values = dict(
        tool_choice=tool_choice,
        include_reasoning=include_reasoning,
        tools=TOOLS,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        include_stop_str_in_output=False,
        parallel_tool_calls=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def normalize_calls(calls):
    return [(call.name, json.loads(call.arguments)) for call in calls or []]


def feed_chunks(
    parser,
    chunks,
    req,
    *,
    prompt_token_ids=PROMPT_TOKEN_IDS,
):
    deltas = []
    for index, chunk in enumerate(chunks):
        delta = parser.parse_delta(
            chunk,
            [],
            req,
            prompt_token_ids=prompt_token_ids if index == 0 else None,
            finished=index == len(chunks) - 1,
        )
        if delta is not None:
            deltas.append(delta)
    return deltas


def collect_stream(
    parser,
    text,
    req,
    chunk_size,
    *,
    prompt_token_ids=PROMPT_TOKEN_IDS,
):
    reasoning = []
    content = []
    calls: dict[int, dict[str, str]] = {}
    for start in range(0, len(text), chunk_size):
        end = min(start + chunk_size, len(text))
        delta = parser.parse_delta(
            text[start:end],
            [],
            req,
            prompt_token_ids=prompt_token_ids if start == 0 else None,
            finished=end == len(text),
        )
        if delta is None:
            continue
        if delta.reasoning:
            reasoning.append(delta.reasoning)
        if delta.content:
            content.append(delta.content)
        for call in delta.tool_calls:
            function = call.function
            assert function is not None
            slot = calls.setdefault(
                call.index,
                {"name": "", "arguments": ""},
            )
            slot["name"] += function.name or ""
            slot["arguments"] += function.arguments or ""
    normalized_calls = [
        (call["name"], json.loads(call["arguments"]))
        for _, call in sorted(calls.items())
    ]
    return "".join(reasoning), "".join(content), normalized_calls


def test_resolve_response_template_from_tokenizer_configuration():
    tokenizer = FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE)
    assert resolve_response_template(tokenizer) == GEMMA4_RESPONSE_TEMPLATE

    tokenizer = FakeTokenizer()
    del tokenizer.response_template
    tokenizer.init_kwargs["response_template"] = GEMMA4_RESPONSE_TEMPLATE
    assert resolve_response_template(tokenizer) == GEMMA4_RESPONSE_TEMPLATE

    explicit = deepcopy(GEMMA4_RESPONSE_TEMPLATE)
    explicit["defaults"]["role"] = "custom"
    assert resolve_response_template(tokenizer, explicit) is explicit


def test_response_template_parsers_are_registered():
    assert (
        ToolParserManager.get_tool_parser("response_template")
        is ResponseTemplateToolParser
    )
    assert (
        ReasoningParserManager.get_reasoning_parser("response_template")
        is ResponseTemplateReasoningParser
    )


@pytest.mark.parametrize(
    ("reasoning", "tools"),
    [(True, True), (True, False), (False, True)],
    ids=["both", "reasoning", "tools"],
)
def test_parser_manager_selects_response_template_by_name(reasoning, tools):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="response_template" if tools else None,
        reasoning_parser_name="response_template" if reasoning else None,
        enable_auto_tools=tools,
        tokenizer=FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
    )

    assert parser_cls is not None
    assert issubclass(parser_cls, ResponseTemplateParser)
    assert (parser_cls.reasoning_parser_cls is not None) is reasoning
    assert (parser_cls.tool_parser_cls is not None) is tools


def test_parser_manager_validates_checkpoint_metadata_at_startup():
    with pytest.raises(TypeError, match="requires `response_template` metadata"):
        ParserManager.get_parser(
            reasoning_parser_name="response_template",
            tokenizer=FakeTokenizer(),
        )
    with pytest.raises(TypeError, match="Invalid response_template"):
        ParserManager.get_parser(
            reasoning_parser_name="response_template",
            tokenizer=FakeTokenizer({"fields": {}}),
        )

    template = deepcopy(GEMMA4_RESPONSE_TEMPLATE)
    del template["fields"]["tool_calls"]
    with pytest.raises(TypeError, match="tool_calls field"):
        ParserManager.get_parser(
            tool_parser_name="response_template",
            enable_auto_tools=True,
            tokenizer=FakeTokenizer(template),
        )
    assert (
        ParserManager.get_parser(
            reasoning_parser_name="response_template",
            tokenizer=FakeTokenizer(template),
        )
        is not None
    )


def test_response_template_parser_rejects_other_parsers():
    with pytest.raises(TypeError, match="cannot be combined"):
        ParserManager.get_parser(
            tool_parser_name="hermes",
            reasoning_parser_name="response_template",
            enable_auto_tools=True,
            tokenizer=FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        )


def test_request_chat_template_is_honored_with_warning(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        response_template_module.logger, "warning_once", warnings.append
    )
    parser = ResponseTemplateParser(FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE), TOOLS)

    parser.adjust_request(request(chat_template="{{ messages }}"))

    assert len(warnings) == 1
    assert parser.response_template.fields


def test_reasoning_gate_waits_for_content_region():
    parser = ResponseTemplateParser(
        FakeTokenizer(
            GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + "<|channel>thought\n",
        ),
        TOOLS,
    )

    assert parser.is_reasoning_end([1]) is False
    assert parser.is_reasoning_end([2]) is True


def test_reasoning_gate_ends_at_content_opener_held_by_prompt():
    template = deepcopy(GEMMA4_RESPONSE_TEMPLATE)
    template["fields"]["content"]["open_pattern"] = r"<\|channel>final\n"
    parser = ResponseTemplateParser(
        FakeTokenizer(template, prefix=PREFIX + "<|channel>final\n"),
        TOOLS,
    )

    assert parser.is_reasoning_end([1]) is True


def test_reasoning_gate_honors_disabled_and_repeated_thinking():
    disabled = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert disabled.is_reasoning_end([1]) is True

    repeated_template = deepcopy(GEMMA4_RESPONSE_TEMPLATE)
    repeated_template["fields"]["thinking"]["repeats"] = True
    repeated = ResponseTemplateParser(
        FakeTokenizer(repeated_template),
        TOOLS,
        response_template=repeated_template,
    )
    assert repeated.is_reasoning_end(OPEN_REASONING_PROMPT_TOKEN_IDS) is False
    assert repeated.is_reasoning_end([2]) is False


@pytest.mark.parametrize(
    "text",
    [
        "Hello<turn|>",
        THINKING + "The answer is 42.<turn|>",
        THINKING + TOOL_GENERATION,
        "Let me check." + TOOL_GENERATION,
        (
            '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,'
            'unit:<|"|>celsius<|"|>}<tool_call|>' + TOOL_GENERATION
        ),
        (
            "<|tool_call>call:complex_function{"
            'nested:{inner:<|"|>value<|"|>},'
            'list:[<|"|>a<|"|>,<|"|>b<|"|>]}'
            "<tool_call|>"
        ),
    ],
    ids=[
        "content",
        "reasoning",
        "tool",
        "content_then_tool",
        "multiple_tools",
        "nested_arguments",
    ],
)
def test_gemma4_response_template_matches_registered_parser(text):
    tokenizer = FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE)
    req = request()
    registered = Gemma4Parser(tokenizer, TOOLS)
    metadata = ResponseTemplateParser(
        tokenizer,
        TOOLS,
        response_template=GEMMA4_RESPONSE_TEMPLATE,
    )
    metadata.set_prompt_token_ids([1])

    expected_reasoning, expected_content, expected_calls = registered.parse(
        text,
        req,
        enable_auto_tools=True,
    )
    reasoning, content, calls = metadata.parse(
        text,
        req,
        enable_auto_tools=True,
    )

    assert reasoning == expected_reasoning
    assert content == expected_content
    assert normalize_calls(calls) == normalize_calls(expected_calls)


def test_metadata_parser_keeps_already_decoded_nullable_values():
    text = (
        "<|tool_call>call:configure{"
        "enabled:false,ratio:5.0,label:null,value:null}"
        "<tool_call|>"
    )
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        response_template=GEMMA4_RESPONSE_TEMPLATE,
    )
    parser.set_prompt_token_ids([1])

    _, content, calls = parser.parse(
        text,
        request(),
        enable_auto_tools=True,
    )

    assert content is None
    assert normalize_calls(calls) == [
        (
            "configure",
            {
                "enabled": False,
                "ratio": 5.0,
                "label": None,
                "value": None,
            },
        )
    ]


@pytest.mark.parametrize(
    "text",
    [
        THINKING + TOOL_GENERATION,
        THINKING + "The answer is 42.<turn|>",
        "Let me check. " + TOOL_GENERATION,
        (
            '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
            + TOOL_GENERATION
        ),
    ],
    ids=[
        "reasoning_then_tool",
        "reasoning_then_content",
        "content_then_tool",
        "multiple_tools",
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 7, 10_000])
def test_gemma4_response_template_streaming_matches_registered_parser(
    text,
    chunk_size,
):
    req = request()
    registered = Gemma4Parser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
    )
    metadata = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )

    assert collect_stream(metadata, text, req, chunk_size) == collect_stream(
        registered,
        text,
        req,
        chunk_size,
    )


@pytest.mark.parametrize(
    ("prompt_token_ids", "text"),
    [
        (
            OPEN_REASONING_PROMPT_TOKEN_IDS,
            "continued reasoning<channel|>Final answer<turn|>",
        ),
        (
            PROMPT_TOKEN_IDS,
            "Direct answer without reasoning.<turn|>",
        ),
    ],
    ids=["prefilled_reasoning", "direct_answer"],
)
def test_gemma4_response_template_prompt_state_matches_registered_parser(
    prompt_token_ids,
    text,
):
    req = request()
    registered = Gemma4Parser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        chat_template_kwargs={"enable_thinking": True},
    )
    metadata = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )

    assert collect_stream(
        metadata,
        text,
        req,
        1,
        prompt_token_ids=prompt_token_ids,
    ) == collect_stream(
        registered,
        text,
        req,
        1,
        prompt_token_ids=prompt_token_ids,
    )


def test_gemma4_response_template_reasoning_suppression_matches_registered_parser():
    req = request(include_reasoning=False)
    text = THINKING + "Final answer<turn|>"

    assert collect_stream(
        ResponseTemplateParser(
            FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
            TOOLS,
            enable_auto_tools=True,
        ),
        text,
        req,
        1,
    ) == collect_stream(
        Gemma4Parser(FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE), TOOLS),
        text,
        req,
        1,
    )


def test_gemma4_response_template_streams_name_before_validated_arguments():
    chunks = [
        "<|tool_call>call:set_alarm{",
        "hour:7,",
        'label:<|"|>morning<|"|>}',
        "<tool_call|>",
    ]
    req = request()
    registered_deltas = feed_chunks(
        Gemma4Parser(FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE), TOOLS),
        chunks,
        req,
    )
    metadata_deltas = feed_chunks(
        ResponseTemplateParser(
            FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
            TOOLS,
            enable_auto_tools=True,
        ),
        chunks,
        req,
    )

    assert any(delta.tool_calls for delta in registered_deltas[:-1])
    early_calls = [call for delta in metadata_deltas[:-1] for call in delta.tool_calls]
    assert len(early_calls) == 1
    assert early_calls[0].id is not None
    assert early_calls[0].index == 0
    assert early_calls[0].function is not None
    assert early_calls[0].function.name == "set_alarm"
    assert early_calls[0].function.arguments is None

    closing_calls = metadata_deltas[-1].tool_calls
    assert len(closing_calls) == 1
    assert closing_calls[0].id is None
    assert closing_calls[0].index == 0
    assert closing_calls[0].function is not None
    assert closing_calls[0].function.name is None
    assert json.loads(closing_calls[0].function.arguments) == {
        "hour": 7,
        "label": "morning",
    }


def test_response_template_coalesces_name_and_arguments_from_one_chunk():
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )

    delta = parser.parse_delta(
        TOOL_GENERATION,
        [],
        request(),
        prompt_token_ids=PROMPT_TOKEN_IDS,
        finished=True,
    )

    assert delta is not None
    assert len(delta.tool_calls) == 1
    call = delta.tool_calls[0]
    assert call.id is not None
    assert call.function is not None
    assert call.function.name == "set_alarm"
    assert json.loads(call.function.arguments) == {"hour": 7, "label": "morning"}


def test_response_template_defers_name_derived_from_region_content():
    template = {
        "defaults": {"role": "assistant"},
        "start_anchor": PREFIX,
        "fields": {
            "tool_calls": {
                "open": "<tool>",
                "close": "</tool>",
                "content": "json",
                "transform": {
                    "type": "function",
                    "function": {
                        "name": "{content.name}",
                        "arguments": "{content.arguments}",
                    },
                },
            }
        },
    }
    chunks = [
        "<tool>",
        '{"name":"set_',
        'alarm","arguments":{"hour":7}}',
        "</tool>",
    ]
    parser = ResponseTemplateParser(
        FakeTokenizer(template),
        TOOLS,
        response_template=template,
        enable_auto_tools=True,
    )

    deltas = feed_chunks(parser, chunks, request())

    assert all(not delta.tool_calls for delta in deltas[:-1])
    assert len(deltas[-1].tool_calls) == 1
    call = deltas[-1].tool_calls[0]
    assert call.function is not None
    assert call.function.name == "set_alarm"
    assert json.loads(call.function.arguments) == {"hour": 7}


def test_response_template_streams_literal_transformed_name():
    template = {
        "defaults": {"role": "assistant"},
        "start_anchor": PREFIX,
        "fields": {
            "tool_calls": {
                "open": "<call>",
                "close": "</call>",
                "content": "json",
                "transform": {
                    "type": "function",
                    "function": {
                        "name": "set_alarm",
                        "arguments": "{content}",
                    },
                },
            }
        },
    }
    parser = ResponseTemplateParser(
        FakeTokenizer(template),
        TOOLS,
        response_template=template,
        enable_auto_tools=True,
    )

    deltas = feed_chunks(
        parser,
        ["<call>", '{"hour":7}', "</call>"],
        request(),
    )

    assert deltas[0].tool_calls[0].function.name == "set_alarm"
    assert deltas[0].tool_calls[0].function.arguments is None
    assert deltas[-1].tool_calls[0].function.name is None
    assert json.loads(deltas[-1].tool_calls[0].function.arguments) == {"hour": 7}


def test_gemma4_response_template_drops_incomplete_call():
    text = '<|tool_call>call:get_weather{city:<|"|>London'
    req = request()
    registered = Gemma4Parser(FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE), TOOLS)
    metadata = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
    )
    metadata.set_prompt_token_ids(PROMPT_TOKEN_IDS)

    _, _, registered_calls = registered.parse(
        text,
        req,
        enable_auto_tools=True,
    )
    _, metadata_content, metadata_calls = metadata.parse(
        text,
        req,
        enable_auto_tools=True,
    )

    assert normalize_calls(registered_calls) == [("get_weather", {"city": "London"})]
    assert metadata_content is None
    assert metadata_calls is None


def test_gemma4_response_template_preserves_content_after_tool_call():
    text = TOOL_GENERATION + "Done.<turn|>"
    req = request()
    registered = Gemma4Parser(FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE), TOOLS)
    metadata = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
    )
    metadata.set_prompt_token_ids(PROMPT_TOKEN_IDS)

    _, registered_content, registered_calls = registered.parse(
        text,
        req,
        enable_auto_tools=True,
    )
    _, metadata_content, metadata_calls = metadata.parse(
        text,
        req,
        enable_auto_tools=True,
    )

    assert registered_content is None
    assert metadata_content == "Done."
    assert normalize_calls(metadata_calls) == normalize_calls(registered_calls)


def test_gemma4_checkpoint_grammar_excludes_hyphenated_name():
    text = '<|tool_call>call:get-weather{city:<|"|>London<|"|>}<tool_call|>'
    req = request()
    registered = Gemma4Parser(FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE), TOOLS)
    metadata = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
    )
    metadata.set_prompt_token_ids(PROMPT_TOKEN_IDS)

    _, _, registered_calls = registered.parse(
        text,
        req,
        enable_auto_tools=True,
    )
    _, metadata_content, metadata_calls = metadata.parse(
        text,
        req,
        enable_auto_tools=True,
    )

    assert normalize_calls(registered_calls) == [("get-weather", {"city": "London"})]
    assert metadata_content is None
    assert metadata_calls is None


@pytest.mark.parametrize(
    (
        "prefix_suffix",
        "generated",
        "enable_auto_tools",
        "expected_reasoning",
        "expected_content",
    ),
    [
        (
            "<|channel>thought\nprefill",
            " generated<channel|>",
            False,
            " generated",
            None,
        ),
        ("prefill ", "generated<turn|>", False, None, "generated"),
        ("prefill", "", False, None, None),
        (TOOL_GENERATION.split("{", 1)[0], "", True, None, None),
        ("<|chan", "", False, None, None),
        ("<|chan", "nel", False, None, "nel"),
        (
            "<|chan",
            "nel>thought\nreason<channel|>",
            False,
            "reason",
            None,
        ),
    ],
    ids=[
        "reasoning",
        "content",
        "empty_content",
        "empty_tool_open",
        "empty_partial_delimiter",
        "unresolved_partial_delimiter",
        "completed_partial_delimiter",
    ],
)
def test_streaming_prefill_is_state_only(
    prefix_suffix,
    generated,
    enable_auto_tools,
    expected_reasoning,
    expected_content,
):
    tokenizer = FakeTokenizer(
        GEMMA4_RESPONSE_TEMPLATE,
        prefix=PREFIX + prefix_suffix,
    )
    parser = ResponseTemplateParser(
        tokenizer,
        TOOLS,
        enable_auto_tools=enable_auto_tools,
    )

    delta = parser.parse_delta(
        generated,
        [],
        request(),
        prompt_token_ids=[1],
        finished=True,
    )

    if expected_reasoning is None and expected_content is None:
        assert delta is None
    else:
        assert delta is not None
        assert delta.reasoning == expected_reasoning
        assert delta.content == expected_content


def test_malformed_tool_opened_across_prompt_boundary_stays_incomplete():
    parser = ResponseTemplateParser(
        FakeTokenizer(
            GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + "<|tool_call>call:set_",
        ),
        TOOLS,
        enable_auto_tools=True,
    )

    opened = parser.parse_delta(
        "alarm{",
        [],
        request(),
        prompt_token_ids=[1],
        finished=False,
    )
    delta = parser.parse_delta(
        "hour:7}unexpected<tool_call|>",
        [],
        request(),
        finished=True,
    )

    assert opened is not None
    assert delta is None
    functions = [call.function for call in opened.tool_calls if call.function]
    assert "".join(function.name or "" for function in functions) == "set_alarm"
    assert "".join(function.arguments or "" for function in functions) == ""
    assert parser.incomplete_tool_call_indices == {0}


def test_responses_namespace_tools_use_vllm_flattened_names():
    namespace = NamespaceTool(
        type="namespace",
        name="calendar",
        description="Calendar tools",
        tools=[
            {
                "type": "function",
                "name": "set_alarm",
                "description": "Set an alarm",
                "parameters": TOOLS[0].function.parameters,
            }
        ],
    )
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        [namespace],
    )
    parser.set_prompt_token_ids([1])
    output = TOOL_GENERATION.replace(
        "call:set_alarm",
        "call:calendar__set_alarm",
    )

    _, content, calls = parser.parse(
        output,
        request(),
        enable_auto_tools=True,
    )

    assert content is None
    assert normalize_calls(calls) == [
        ("calendar__set_alarm", {"hour": 7, "label": "morning"})
    ]


def test_streaming_unknown_call_is_forwarded():
    text = TOOL_GENERATION.replace("call:set_alarm", "call:unknown")
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )
    _, content, calls = collect_stream(parser, text, request(), 7)

    assert content == ""
    assert calls == [("unknown", {"hour": 7, "label": "morning"})]


@pytest.mark.parametrize(
    "text",
    [
        TOOL_GENERATION.replace(
            "<tool_call|>",
            "unexpected<tool_call|>",
        ),
        TOOL_GENERATION.removesuffix("<tool_call|>") + "<tool_",
    ],
    ids=["malformed", "truncated"],
)
def test_streaming_invalid_call_leaves_started_tool_call_incomplete(text):
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )
    deltas = feed_chunks(
        parser,
        [text[start : start + 7] for start in range(0, len(text), 7)],
        request(),
    )

    assert "".join(delta.content or "" for delta in deltas) == ""
    tool_calls = [call for delta in deltas for call in delta.tool_calls]
    assert {call.index for call in tool_calls} == {0}
    functions = [call.function for call in tool_calls]
    assert all(function is not None for function in functions)
    assert "".join(function.name or "" for function in functions) == "set_alarm"
    assert "".join(function.arguments or "" for function in functions) == ""
    assert parser.incomplete_tool_call_indices == {0}


def test_closed_malformed_call_does_not_block_following_call():
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )
    opening, body = TOOL_GENERATION.split("{", 1)
    first = parser.parse_delta(
        opening + "{",
        [],
        request(),
        prompt_token_ids=[1],
        finished=False,
    )
    second = parser.parse_delta(
        body.replace("<tool_call|>", "unexpected<tool_call|>") + TOOL_GENERATION,
        [],
        request(),
        finished=True,
    )

    assert first is not None
    assert second is not None
    calls = [call for delta in (first, second) for call in delta.tool_calls]
    assert {call.index for call in calls} == {0, 1}
    assert parser.incomplete_tool_call_indices == {0}
    second_arguments = "".join(
        call.function.arguments or ""
        for call in calls
        if call.index == 1 and call.function is not None
    )
    assert json.loads(second_arguments) == {
        "hour": 7,
        "label": "morning",
    }


def test_non_streaming_malformed_call_is_dropped():
    text = TOOL_GENERATION.replace(
        "<tool_call|>",
        "unexpected<tool_call|>",
    )
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
    )
    parser.set_prompt_token_ids([1])

    reasoning, content, calls = parser.parse(
        text,
        request(),
        enable_auto_tools=True,
    )

    assert reasoning is None
    assert content is None
    assert calls is None


def test_streaming_malformed_call_is_dropped_before_emission():
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )
    first = parser.parse_delta(
        "hello<turn|>",
        [],
        request(),
        prompt_token_ids=[1],
        finished=False,
    )
    malformed = TOOL_GENERATION.replace(
        "<tool_call|>",
        "unexpected<tool_call|>",
    )
    failed = parser.parse_delta(
        malformed,
        [],
        request(),
        finished=False,
    )

    assert first is not None
    assert first.content == "hello"
    assert failed is None


def test_prefilled_malformed_call_is_dropped():
    opening, body = TOOL_GENERATION.split("{", 1)
    malformed_body = "{" + body.replace(
        "<tool_call|>",
        "unexpected<tool_call|>",
    )
    tokenizer = FakeTokenizer(
        GEMMA4_RESPONSE_TEMPLATE,
        prefix=PREFIX + opening,
    )
    parser = ResponseTemplateParser(
        tokenizer,
        TOOLS,
        enable_auto_tools=True,
    )

    delta = parser.parse_delta(
        malformed_body,
        [],
        request(),
        prompt_token_ids=[1],
        finished=True,
    )

    assert delta is None
    assert not parser.has_incomplete_tool_call


def test_tool_choice_none_preserves_raw_call_as_content():
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )
    _, content, calls = collect_stream(
        parser,
        TOOL_GENERATION,
        request("none"),
        5,
    )

    assert content == TOOL_GENERATION
    assert calls == []


def test_adjust_request_preserves_parser_delimiters_without_forcing_stop_text():
    req = request()
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
    )

    assert parser.adjust_request(req) is req
    assert req.skip_special_tokens is False
    assert req.spaces_between_special_tokens is False
    assert req.include_stop_str_in_output is False


def test_call_without_closer_is_dropped():
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
    )
    parser.set_prompt_token_ids([1])
    trimmed = TOOL_GENERATION.removesuffix("<tool_call|>")

    _, content, calls = parser.parse(
        trimmed,
        request(),
        enable_auto_tools=True,
    )

    assert content is None
    assert calls is None


def test_tool_closer_stop_token_completes_call():
    trimmed = TOOL_GENERATION.removesuffix("<tool_call|>")
    closer_id = VOCAB["<tool_call|>"]
    parser = ResponseTemplateParser(FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE), TOOLS)
    parser.set_prompt_token_ids([1])
    streaming = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )

    _, content, calls = parser.parse(
        trimmed,
        request(),
        enable_auto_tools=True,
        model_output_token_ids=[1, closer_id],
    )
    streamed = streaming.parse_delta(
        trimmed,
        [1, closer_id],
        request(),
        prompt_token_ids=[1],
        finished=True,
    )

    assert content is None
    assert normalize_calls(calls) == [("set_alarm", {"hour": 7, "label": "morning"})]
    assert streamed is not None
    assert [call.function.name for call in streamed.tool_calls] == ["set_alarm"]
    assert not streaming.has_incomplete_tool_call


@pytest.mark.parametrize(
    ("req", "match"),
    [
        (request("required"), "required tool choice"),
        (
            request(
                ChatCompletionNamedToolChoiceParam(
                    type="function",
                    function={"name": "set_alarm"},
                )
            ),
            "named tool choice",
        ),
        (request("auto", parallel_tool_calls=False), "parallel_tool_calls=False"),
        (
            request(
                "auto",
                tools=[
                    {
                        "type": "function",
                        "name": "set_alarm",
                        "parameters": {"type": "object"},
                        "strict": True,
                    }
                ],
            ),
            "strict tools",
        ),
    ],
)
def test_unconstrained_tool_guarantees_are_rejected(req, match):
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        req.tools,
        enable_auto_tools=True,
    )

    with pytest.raises(VLLMValidationError, match=match):
        parser.adjust_request(req)


def test_tool_choice_none_does_not_require_structural_guarantees():
    req = request(
        "none",
        tools=[
            {
                "type": "function",
                "name": "set_alarm",
                "parameters": {"type": "object"},
                "strict": True,
            }
        ],
        parallel_tool_calls=False,
    )
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        req.tools,
        enable_auto_tools=True,
    )

    assert parser.adjust_request(req) is req


def test_unsupported_semantic_field_is_rejected():
    template = deepcopy(GEMMA4_RESPONSE_TEMPLATE)
    template["fields"]["citations"] = {
        "open": "<citation>",
        "close": "</citation>",
    }

    with pytest.raises(ValueError, match="citations"):
        ResponseTemplateParser(
            FakeTokenizer(template),
            TOOLS,
            response_template=template,
        )


@pytest.mark.parametrize(
    ("field_name", "field_update"),
    [
        ("content", {"content": "json"}),
        ("thinking", {"transform": "{content}"}),
    ],
)
def test_unsupported_content_field_semantics_are_rejected(
    field_name,
    field_update,
):
    template = deepcopy(GEMMA4_RESPONSE_TEMPLATE)
    template["fields"][field_name].update(field_update)

    with pytest.raises(ValueError, match="cannot be streamed"):
        ResponseTemplateParser(
            FakeTokenizer(template),
            TOOLS,
            response_template=template,
        )


def test_delimited_template_keeps_plain_content_and_tolerates_required_fields():
    template = {
        "start_anchor": "<assistant>",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>"},
            "tool_calls": {
                "open": "<call>",
                "close": "</call>",
                "content": "json",
                "optional": False,
            },
        },
    }
    parser = ResponseTemplateParser(
        FakeTokenizer(template, prefix="<assistant>"),
        TOOLS,
        response_template=template,
    )

    assert parser.parse("<think>plan</think>answer", request()) == (
        "plan",
        "answer",
        None,
    )


def test_separate_parse_calls_use_distinct_tool_call_ids():
    parser = ResponseTemplateParser(
        FakeTokenizer(GEMMA4_RESPONSE_TEMPLATE),
        TOOLS,
        enable_auto_tools=True,
    )
    parsed_calls = []
    for _ in range(2):
        _, _, calls = parser.parse(
            TOOL_GENERATION,
            request(),
            enable_auto_tools=True,
        )
        assert calls is not None
        parsed_calls.append(calls[0])

    assert parsed_calls[0].id is not None
    assert parsed_calls[1].id is not None
    assert parsed_calls[0].id != parsed_calls[1].id
