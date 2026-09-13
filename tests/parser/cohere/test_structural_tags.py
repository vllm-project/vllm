# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``adjust_request``: folding tools and JSON schemas into Cohere structural tags."""

from __future__ import annotations

import json
from collections import UserDict
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm.entrypoints.generate.base.protocol import (
    JsonSchemaResponseFormat,
    ResponseFormat,
    StructuralTagResponseFormat,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.cohere_command import (
    CohereCommandParser,
    _has_effective_tools,
    _response_format_type,
    _schema_dict_from_structured_outputs,
    convert_schema_to_structural_tags,
)
from vllm.sampling_params import StructuredOutputsParams

from .utils import MockCohereTokenizer, make_parser

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
def parser(tokenizer: MockCohereTokenizer) -> CohereCommandParser:
    """Parser configured with a supported Cohere architecture."""
    return make_parser(
        tokenizer, "cohere_command4", model_config=_model_config("Cohere2ForCausalLM")
    )


@pytest.fixture(scope="module")
def parser_no_model_config(tokenizer: MockCohereTokenizer) -> CohereCommandParser:
    """Parser with no ``model_config`` (cannot resolve architecture)."""
    return make_parser(tokenizer, "cohere_command4")


@pytest.fixture(scope="module")
def parser_unsupported_arch(tokenizer: MockCohereTokenizer) -> CohereCommandParser:
    """Parser configured with an architecture that has no structural tag style."""
    return make_parser(
        tokenizer, "cohere_command4", model_config=_model_config("LlamaForCausalLM")
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
