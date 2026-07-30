# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections import UserDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    JsonSchemaResponseFormat,
    ResponseFormat,
    StructuralTagResponseFormat,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.cohere_command_reasoning_parser import (
    CohereCommand3ReasoningParser,
    CohereCommand4ReasoningParser,
    _has_effective_tools,
    _response_format_type,
    _schema_dict_from_structured_outputs,
    convert_schema_to_structural_tags,
)
from vllm.sampling_params import StructuredOutputsParams


@dataclass
class ExpectedToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ReasoningCase:
    parser_cls: Any
    model_output: str
    expected_reasoning: str | None
    expected_content: str | None
    expected_tool_calls: list[ExpectedToolCall] = field(default_factory=list)


REASONING_CASES = [
    pytest.param(
        ReasoningCase(
            parser_cls=CohereCommand3ReasoningParser,
            model_output="""\
<|START_THINKING|> i will call foo with query1<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}}
]
<|END_ACTION|>""",
            expected_reasoning="i will call foo with query1",
            expected_content="""\
<|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}}
]
<|END_ACTION|>""",
            expected_tool_calls=[
                ExpectedToolCall(id="0", name="foo", arguments={"query": "query1"}),
            ],
        ),
        id="cmd3-single_tool_call",
    ),
    pytest.param(
        ReasoningCase(
            parser_cls=CohereCommand4ReasoningParser,
            model_output="""\
<|START_THINKING|> i will call foo with query1<|END_THINKING|><|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}}
]
<|END_ACTION|>""",
            expected_reasoning="i will call foo with query1",
            expected_content="""\
<|START_ACTION|>
[
    {"tool_call_id": "0", "tool_name": "foo", "parameters": {"query": "query1"}}
]
<|END_ACTION|>""",
            expected_tool_calls=[
                ExpectedToolCall(id="0", name="foo", arguments={"query": "query1"}),
            ],
        ),
        id="cmd4-single_tool_call",
    ),
    pytest.param(
        ReasoningCase(
            parser_cls=CohereCommand3ReasoningParser,
            model_output="""\
<|START_THINKING|>This is a rainbow <co>emoji: 🌈</co: 0:[1]><|END_THINKING|>
<|START_RESPONSE|>foo <co>bar</co: 0:[1,2],1:[3,4]><|END_RESPONSE|>""",
            expected_reasoning="This is a rainbow emoji: 🌈",
            expected_content="foo bar",
        ),
        id="cmd3-citations_with_emoji",
    ),
    pytest.param(
        ReasoningCase(
            parser_cls=CohereCommand4ReasoningParser,
            model_output="""\
<|START_THINKING|>This is a rainbow <co>emoji: 🌈</co: 0:[1]><|END_THINKING|>
<|START_RESPONSE|>foo <co>bar</co: 0:[1,2],1:[3,4]><|END_RESPONSE|>""",
            expected_reasoning="This is a rainbow emoji: 🌈",
            expected_content="foo bar",
        ),
        id="cmd4-citations_with_emoji",
    ),
]


class MockCohereTokenizer:
    """Minimal byte-level stand-in for the Cohere tokenizer.

    ``encode``/``decode`` round-trip through UTF-8 bytes so splitting a
    multi-byte character (e.g. an emoji) across "tokens" reproduces the
    trailing U+FFFD buffering that real streaming exhibits. Cohere special
    tokens map to distinct synthetic ids; everything else shares a default id.
    ``adjust_request`` only needs the token ids, not real tokenization.
    """

    _SPECIAL_TOKEN_IDS = {
        "<|START_THINKING|>": -1,
        "<|END_THINKING|>": -2,
        "<|CHATBOT_TOKEN|>": -3,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIAL_TOKEN_IDS.get(token, 0)

    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def tokenizer() -> MockCohereTokenizer:
    return MockCohereTokenizer()


@pytest.fixture
def request_obj():
    return ChatCompletionRequest(messages=[], model="test-model")


REPLACEMENT_CHAR = "\ufffd"


def _token_deltas(tokenizer, text: str) -> list[str]:
    """Progressively decode the token sequence and return per-step string
    deltas.  Incomplete multi-byte sequences (trailing U+FFFD) are buffered
    until the next token completes them, matching real streaming behaviour."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    deltas: list[str] = []
    prev = ""
    for i in range(1, len(ids) + 1):
        current = tokenizer.decode(ids[:i], skip_special_tokens=False)
        if current.endswith(REPLACEMENT_CHAR):
            continue
        delta = current[len(prev) :]
        if delta:
            deltas.append(delta)
        prev = current
    return deltas


@pytest.mark.parametrize("case", REASONING_CASES)
class TestExtractReasoning:
    def test_nonstreaming(self, tokenizer, request_obj, case: ReasoningCase):
        parser = case.parser_cls(tokenizer)
        reasoning, content = parser.extract_reasoning(case.model_output, request_obj)

        assert reasoning == case.expected_reasoning
        assert content == case.expected_content

    def test_streaming(self, tokenizer, case: ReasoningCase):
        parser = case.parser_cls(tokenizer)
        token_strings = _token_deltas(tokenizer, case.model_output)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_call_deltas: list[dict] = []

        previous_text = ""
        previous_token_ids: list[int] = []

        for token_str in token_strings:
            current_text = previous_text + token_str
            current_token_ids = previous_token_ids + [0]

            delta = parser.extract_reasoning_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=token_str,
                previous_token_ids=previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=[0],
            )
            if delta is not None:
                if delta.reasoning is not None:
                    reasoning_parts.append(delta.reasoning)
                if delta.content is not None:
                    content_parts.append(delta.content)
                for tc in delta.tool_calls:
                    tool_call_deltas.append(
                        {
                            "id": tc.id,
                            "index": tc.index,
                            "name": tc.function.name if tc.function else None,
                            "arguments": (
                                tc.function.arguments if tc.function else None
                            ),
                        }
                    )

            previous_text = current_text
            previous_token_ids = current_token_ids

        reasoning = "".join(reasoning_parts) if reasoning_parts else None
        assert reasoning == case.expected_reasoning

        content = "".join(content_parts) if content_parts else None
        if case.expected_tool_calls:
            assert content is None or content == ""
        else:
            assert content == case.expected_content

        accumulated: dict[int, dict] = {}
        for d in tool_call_deltas:
            idx = d["index"]
            if idx not in accumulated:
                accumulated[idx] = {"id": "", "name": "", "arguments": ""}
            if d["id"]:
                accumulated[idx]["id"] = d["id"]
            if d["name"]:
                accumulated[idx]["name"] = d["name"]
            if d["arguments"]:
                accumulated[idx]["arguments"] += d["arguments"]

        assert len(accumulated) == len(case.expected_tool_calls)
        for i, expected_tc in enumerate(case.expected_tool_calls):
            tc = accumulated[i]
            assert tc["id"] == expected_tc.id
            assert tc["name"] == expected_tc.name
            assert json.loads(tc["arguments"]) == expected_tc.arguments


class TestIsReasoningEnd:
    @pytest.mark.parametrize(
        "parser_cls",
        [CohereCommand3ReasoningParser, CohereCommand4ReasoningParser],
        ids=["cmd3", "cmd4"],
    )
    def test_is_reasoning_end(self, tokenizer, parser_cls):
        parser = parser_cls(tokenizer)
        start_id = tokenizer.convert_tokens_to_ids("<|START_THINKING|>")
        end_id = tokenizer.convert_tokens_to_ids("<|END_THINKING|>")
        chatbot_id = tokenizer.convert_tokens_to_ids("<|CHATBOT_TOKEN|>")
        content_ids = [99, 100]

        # Generation-only tokens have no chatbot marker, so the whole sequence
        # is considered.
        assert parser.is_reasoning_end([end_id])
        assert parser.is_reasoning_end([start_id, *content_ids, end_id])
        assert not parser.is_reasoning_end([start_id, *content_ids])

        # Full prompt/history tokens are scoped to the latest chatbot marker,
        # so stray thinking tokens from the preamble or previous turns are ignored.
        assert not parser.is_reasoning_end([start_id, end_id, chatbot_id, *content_ids])
        assert parser.is_reasoning_end(
            [start_id, end_id, chatbot_id, start_id, *content_ids, end_id]
        )


SCHEMA_A = {"type": "object", "properties": {"a": {"type": "string"}}}
SCHEMA_B = {"type": "object", "properties": {"b": {"type": "number"}}}
GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
}
VALID_STRUCTURAL_TAG = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "tags": [
            {
                "begin": "<tool>",
                "content": {"type": "any_text"},
                "end": "</tool>",
            }
        ],
        "triggers": ["<tool>"],
    },
}


def _model_config(arch: str) -> SimpleNamespace:
    return SimpleNamespace(
        architecture=arch,
        architectures=[arch],
        hf_text_config=SimpleNamespace(architectures=[arch]),
    )


def _make_chat_request(**kwargs) -> ChatCompletionRequest:
    data = {"messages": [{"role": "user", "content": "hi"}], "model": "m"}
    data.update(kwargs)
    return ChatCompletionRequest.model_validate(data)


def _first_json_schema(tag_json: str) -> dict | None:
    outer = json.loads(tag_json)
    for t in (outer.get("format") or {}).get("tags") or []:
        c = t.get("content") or {}
        if c.get("type") == "json_schema":
            js = c.get("json_schema")
            return js if isinstance(js, dict) else None
    return None


def _content_types(tag_json: str) -> set[str]:
    outer = json.loads(tag_json)
    out: set[str] = set()
    for t in (outer.get("format") or {}).get("tags") or []:
        ty = (t.get("content") or {}).get("type")
        if isinstance(ty, str):
            out.add(ty)
    return out


@pytest.fixture(scope="module")
def parser(tokenizer: MockCohereTokenizer) -> CohereCommand4ReasoningParser:
    """Parser configured with a supported Cohere architecture."""
    return CohereCommand4ReasoningParser(
        tokenizer,
        model_config=_model_config("Cohere2ForCausalLM"),
    )


@pytest.fixture(scope="module")
def parser_no_model_config(
    tokenizer: MockCohereTokenizer,
) -> CohereCommand4ReasoningParser:
    """Parser with no ``model_config`` (cannot resolve architecture)."""
    return CohereCommand4ReasoningParser(tokenizer, model_config=None)


@pytest.fixture(scope="module")
def parser_unsupported_arch(
    tokenizer: MockCohereTokenizer,
) -> CohereCommand4ReasoningParser:
    """Parser configured with an architecture that has no structural tag style."""
    return CohereCommand4ReasoningParser(
        tokenizer,
        model_config=_model_config("LlamaForCausalLM"),
    )


class TestAdjustRequestPassthrough:
    def test_structured_outputs_structural_tag_not_modified(self, parser) -> None:
        tag = json.dumps(VALID_STRUCTURAL_TAG)
        r = _make_chat_request(structured_outputs={"structural_tag": tag})
        o = parser.adjust_request(r)
        assert o.structured_outputs.structural_tag == tag

    def test_response_format_structural_tag_short_circuit(self, parser) -> None:
        # ``ChatCompletionRequest`` validates ``response_format`` as a union;
        # bare ``{"type": "structural_tag"}`` is invalid (use pydantic model).
        rf = StructuralTagResponseFormat(
            type="structural_tag",
            format=VALID_STRUCTURAL_TAG["format"],
        )
        r = _make_chat_request(response_format=rf)
        o = parser.adjust_request(r)
        assert _response_format_type(o.response_format) == "structural_tag"
        assert o.structured_outputs is None


class TestAdjustRequestNoOp:
    def test_no_schema_no_tools(self, parser) -> None:
        o = parser.adjust_request(_make_chat_request())
        assert o.structured_outputs is None
        assert o.response_format is None

    def test_no_model_config(self, parser_no_model_config) -> None:
        inner = JsonSchemaResponseFormat(name="n", json_schema=SCHEMA_A)
        r = _make_chat_request(
            response_format=ResponseFormat(type="json_schema", json_schema=inner),
        )
        o = parser_no_model_config.adjust_request(r)
        assert o.response_format is not None
        assert o.structured_outputs is None


class TestAdjustRequestUnsupportedArchitecture:
    def test_json_schema_raises(self, parser_unsupported_arch) -> None:
        inner = JsonSchemaResponseFormat(name="n", json_schema=SCHEMA_A)
        r = _make_chat_request(
            response_format=ResponseFormat(type="json_schema", json_schema=inner),
        )
        with pytest.raises(ValueError, match="does not support"):
            parser_unsupported_arch.adjust_request(r)


class TestAdjustRequestFoldFromResponseFormat:
    @pytest.mark.parametrize(
        "response_format, expected_schema",
        [
            pytest.param(
                ResponseFormat(
                    type="json_schema",
                    json_schema=JsonSchemaResponseFormat(
                        name="n", json_schema=SCHEMA_A
                    ),
                ),
                SCHEMA_A,
                id="json_schema_pydantic",
            ),
            pytest.param(
                {
                    "type": "json_schema",
                    "json_schema": {"name": "n", "schema": SCHEMA_A},
                },
                SCHEMA_A,
                id="json_schema_dict",
            ),
            pytest.param(
                {"type": "json_object"},
                {"type": "object"},
                id="json_object",
            ),
        ],
    )
    def test_response_format_cleared(
        self, parser, response_format, expected_schema
    ) -> None:
        r = _make_chat_request(response_format=response_format)
        o = parser.adjust_request(r)
        assert o.response_format is None
        assert (
            _first_json_schema(o.structured_outputs.structural_tag) == expected_schema
        )


class TestHasEffectiveTools:
    @pytest.mark.parametrize(
        "tools, expected",
        [
            pytest.param(None, False, id="none"),
            pytest.param([], False, id="empty_list"),
            pytest.param("   ", False, id="blank_str"),
            pytest.param(
                [{"type": "function", "function": {"name": "f"}}],
                True,
                id="non_empty_list",
            ),
            pytest.param('{"x": 1}', True, id="non_empty_str"),
        ],
    )
    def test_has_effective_tools(self, tools, expected) -> None:
        assert _has_effective_tools(tools) is expected

    def test_convert_schema_json_only_with_empty_tools_list(self) -> None:
        tag = convert_schema_to_structural_tags(
            schema=SCHEMA_B,
            tools=[],
            model_architecture="Cohere2ForCausalLM",
        )
        assert tag is not None
        assert _first_json_schema(tag) == SCHEMA_B


class TestAdjustRequestFoldFromStructuredOutputs:
    @pytest.mark.parametrize(
        "structured_outputs, expected_schema",
        [
            pytest.param({"json": SCHEMA_B}, SCHEMA_B, id="json_dict"),
            pytest.param({"json": json.dumps(SCHEMA_B)}, SCHEMA_B, id="json_string"),
            pytest.param(
                {"json_object": True}, {"type": "object"}, id="json_object_flag"
            ),
            pytest.param(
                StructuredOutputsParams(json=SCHEMA_B),
                SCHEMA_B,
                id="structured_outputs_dataclass",
            ),
            pytest.param(
                {"json": {"name": "n", "schema": SCHEMA_A}},
                SCHEMA_A,
                id="openai_wrapper_dict_unwrapped",
            ),
        ],
    )
    def test_structured_outputs_folded(
        self, parser, structured_outputs, expected_schema
    ) -> None:
        o = parser.adjust_request(
            _make_chat_request(structured_outputs=structured_outputs),
        )
        assert (
            _first_json_schema(o.structured_outputs.structural_tag) == expected_schema
        )

    def test_responses_request_default_empty_tools(self, parser) -> None:
        """``ResponsesRequest.tools`` defaults to ``[]``, not ``None``."""
        r = ResponsesRequest.model_validate(
            {
                "input": "hi",
                "model": "m",
                "structured_outputs": {"json": SCHEMA_B},
            }
        )
        assert r.tools == []
        o = parser.adjust_request(r)
        assert _first_json_schema(o.structured_outputs.structural_tag) == SCHEMA_B

    def test_json_userdict_mapping_unwrapped(self) -> None:
        inner = {"type": "object", "properties": {"u": {"type": "number"}}}
        so = StructuredOutputsParams(json=UserDict(inner))
        assert _schema_dict_from_structured_outputs(so) == inner

    @pytest.mark.parametrize(
        "json_value, match",
        [
            pytest.param("{not json}", "valid JSON", id="invalid_json_string"),
            pytest.param(
                json.dumps(["a", "b"]), "JSON object", id="non_object_json_string"
            ),
            pytest.param("   ", "empty", id="empty_json_string"),
        ],
    )
    def test_structured_outputs_json_string_raises(
        self, parser, json_value, match
    ) -> None:
        with pytest.raises(ValueError, match=match):
            parser.adjust_request(
                _make_chat_request(structured_outputs={"json": json_value}),
            )

    @pytest.mark.parametrize(
        "construct",
        [
            pytest.param(
                lambda: _make_chat_request(structured_outputs={"json": [1, 2, 3]}),
                id="chat_completion_request",
            ),
            pytest.param(
                lambda: StructuredOutputsParams(json=[1, 2, 3]),  # type: ignore[arg-type]
                id="structured_outputs_params",
            ),
        ],
    )
    def test_json_wrong_type_raises(self, construct) -> None:
        """Non-str / non-dict ``json`` fails at Pydantic validation."""
        with pytest.raises(ValidationError):
            construct()


class TestAdjustRequestPrecedence:
    def test_response_format_over_structured_outputs_json(self, parser) -> None:
        s_rf = {"type": "object", "properties": {"rf": {"type": "string"}}}
        s_so = {"type": "object", "properties": {"so": {"type": "number"}}}
        inner = JsonSchemaResponseFormat(name="n", json_schema=s_rf)
        r = _make_chat_request(
            response_format=ResponseFormat(type="json_schema", json_schema=inner),
            structured_outputs={"json": s_so},
        )
        o = parser.adjust_request(r)
        assert _first_json_schema(o.structured_outputs.structural_tag) == s_rf


class TestAdjustRequestTextPlusStructuredOutputs:
    def test_text_response_format_preserved(self, parser) -> None:
        sch = {"type": "object", "properties": {"k": {"type": "string"}}}
        r = _make_chat_request(
            response_format=ResponseFormat(type="text"),
            structured_outputs={"json": sch},
        )
        o = parser.adjust_request(r)
        assert o.response_format is not None
        assert o.response_format.type == "text"
        assert _first_json_schema(o.structured_outputs.structural_tag) == sch


class TestAdjustRequestTools:
    def test_tools_only_command_a_grammar(self, parser) -> None:
        o = parser.adjust_request(
            _make_chat_request(tools=[GET_WEATHER_TOOL], tool_choice="auto"),
        )
        assert "grammar" in _content_types(o.structured_outputs.structural_tag)

    def test_tools_plus_json_schema_both_kinds(self, parser) -> None:
        inner = JsonSchemaResponseFormat(
            name="n",
            json_schema={"type": "object", "properties": {"r": {"type": "string"}}},
        )
        r = _make_chat_request(
            response_format=ResponseFormat(type="json_schema", json_schema=inner),
            tools=[GET_WEATHER_TOOL],
            tool_choice="auto",
        )
        o = parser.adjust_request(r)
        types = _content_types(o.structured_outputs.structural_tag)
        assert "grammar" in types
        assert "json_schema" in types
