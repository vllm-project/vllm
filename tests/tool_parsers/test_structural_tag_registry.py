# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from xgrammar import Grammar, StructuralTag
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.deepseekv3_tool_parser import DeepSeekV3ToolParser
from vllm.tool_parsers.deepseekv4_engine_tool_parser import DeepSeekV4EngineToolParser
from vllm.tool_parsers.deepseekv31_tool_parser import DeepSeekV31ToolParser
from vllm.tool_parsers.deepseekv32_engine_tool_parser import (
    DeepSeekV32EngineToolParser,
)
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser
from vllm.tool_parsers.kimi_k3_tool_parser import KimiK3ToolParser
from vllm.tool_parsers.llama_tool_parser import Llama3JsonToolParser
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.structural_tag_registry import (
    SUPPORTED_STRUCTURAL_TAG_MODELS,
    VLLM_BUILTIN_STRUCTURAL_TAG_MODELS,
    XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS,
    get_function_parameters,
    get_model_structural_tag,
)


@pytest.fixture
def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
    ]


@pytest.fixture
def sample_tools_strict() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
    ]


def test_supported_structural_tag_models_include_vllm_builtins():
    assert SUPPORTED_STRUCTURAL_TAG_MODELS == (
        XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS | VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
    )
    assert "hermes" in VLLM_BUILTIN_STRUCTURAL_TAG_MODELS


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_get_model_structural_tag_supports_all_xgrammar_builtins(
    model: str,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools_strict,
        tool_choice="auto",
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)


def test_get_model_structural_tag_supports_vllm_hermes(
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model="hermes",
        tools=sample_tools,
        tool_choice="required",
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)

    # Assert the semantically meaningful structure rather than the full
    # model_dump(), which gains version-specific keys across xgrammar releases
    # (e.g. "any_order" was added to json_schema content in 0.2.3).
    dump = tag.model_dump()
    assert dump["type"] == "structural_tag"

    fmt = dump["format"]
    assert fmt["type"] == "tags_with_separator"
    assert fmt["separator"] == ""
    assert fmt["at_least_one"] is True
    assert fmt["stop_after_first"] is False

    expected_schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    expected_tags = [
        ('<tool_call>\n{"name": "get_weather", "arguments": ', "}\n</tool_call>"),
        ('<tool_call>{"name": "get_weather", "arguments": ', "}</tool_call>"),
    ]
    assert len(fmt["tags"]) == len(expected_tags)
    for tag_dump, (begin, end) in zip(fmt["tags"], expected_tags):
        assert tag_dump["type"] == "tag"
        assert tag_dump["begin"] == begin
        assert tag_dump["end"] == end
        content = tag_dump["content"]
        assert content["type"] == "json_schema"
        assert content["json_schema"] == expected_schema


def test_hermes_required_tool_calls_use_empty_separator():
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
    ]

    tag = get_model_structural_tag(
        model="hermes",
        tools=tools,
        tool_choice="required",
        reasoning=False,
    )

    assert tag is not None
    assert tag.format.separator == ""


# ---------------------------------------------------------------------------
# Kimi K3 (XTML channel format) structural tag
# ---------------------------------------------------------------------------
_K3_RESPONSE_OPEN = "<|open|>response<|sep|>"
_K3_RESPONSE_CLOSE = "<|close|>response<|sep|>"
_K3_TOOLS_OPEN = "<|open|>tools<|sep|>"
_K3_TOOLS_CLOSE = "<|close|>tools<|sep|>"
_K3_CALL_CLOSE = "<|close|>call<|sep|>"
_K3_ARG_CLOSE = "<|close|>argument<|sep|>"
_K3_MESSAGE_CLOSE = "<|close|>message<|sep|>"


def _k3_tools_by_name() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                    "required": ["city"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "run_command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        ),
    ]


def _k3_arg(key: str, typ: str, val: str) -> str:
    return f'<|open|>argument key="{key}" type="{typ}"<|sep|>{val}{_K3_ARG_CLOSE}'


def _k3_call(name: str, args: str, idx: int = 1) -> str:
    return f'<|open|>call tool="{name}" index="{idx}"<|sep|>{args}{_K3_CALL_CLOSE}'


def _k3_response(content: str = "") -> str:
    return f"{_K3_RESPONSE_OPEN}{content}{_K3_RESPONSE_CLOSE}"


def _k3_tools(*calls: str) -> str:
    return f"{_K3_TOOLS_OPEN}{''.join(calls)}{_K3_TOOLS_CLOSE}"


def _k3_grammar(tool_choice, tools=None):
    tag = get_model_structural_tag(
        model="kimi_k3",
        tools=tools if tools is not None else _k3_tools_by_name(),
        tool_choice=tool_choice,
        reasoning=False,
    )
    assert isinstance(tag, StructuralTag)
    return Grammar.from_structural_tag(tag)


def test_kimi_k3_registered_as_vllm_builtin():
    assert "kimi_k3" in VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
    assert KimiK3ToolParser.structural_tag_model == "kimi_k3"


def test_kimi_k3_auto_without_strict_is_unconstrained():
    # auto + no strict tool => no structural tag (matches the strict gate).
    tag = get_model_structural_tag(
        model="kimi_k3",
        tools=_k3_tools_by_name(),
        tool_choice="auto",
        reasoning=False,
    )
    assert tag is None


@pytest.mark.parametrize(
    "body",
    [
        # single required arg
        _k3_response()
        + _k3_tools(_k3_call("get_weather", _k3_arg("city", "string", "Paris"))),
        # response content + two args (string + number)
        _k3_response("Checking.")
        + _k3_tools(
            _k3_call(
                "get_weather",
                _k3_arg("city", "string", "Paris") + _k3_arg("days", "number", "3"),
            )
        ),
        # args in reverse order (parser is order-agnostic)
        _k3_response()
        + _k3_tools(
            _k3_call(
                "get_weather",
                _k3_arg("days", "number", "3") + _k3_arg("city", "string", "Paris"),
            )
        ),
        # two calls, second tool
        _k3_response()
        + _k3_tools(
            _k3_call("get_weather", _k3_arg("city", "string", "Paris"), 1),
            _k3_call("run_command", _k3_arg("command", "string", "ls -la"), 2),
        ),
        # string value with regex metacharacters / spaces
        _k3_response()
        + _k3_tools(
            _k3_call(
                "run_command", _k3_arg("command", "string", "grep -E 'a|b{2,}' x.py")
            )
        ),
        # trailing message-close marker (model's natural turn terminator)
        _k3_response()
        + _k3_tools(_k3_call("get_weather", _k3_arg("city", "string", "Paris")))
        + _K3_MESSAGE_CLOSE,
        # non-thinking mode: response-open is the prompt prefix, so it is absent
        _K3_RESPONSE_CLOSE
        + _k3_tools(_k3_call("get_weather", _k3_arg("city", "string", "Paris"))),
    ],
)
def test_kimi_k3_required_accepts_valid_tool_calls(body: str):
    assert _is_grammar_accept_string(_k3_grammar("required"), body)


@pytest.mark.parametrize(
    "body",
    [
        # unknown tool name
        _k3_response()
        + _k3_tools(_k3_call("get_temperature", _k3_arg("city", "string", "x"))),
        # number arg given a non-numeric JSON value
        _k3_response()
        + _k3_tools(_k3_call("get_weather", _k3_arg("days", "number", "abc"))),
        # undeclared argument key
        _k3_response()
        + _k3_tools(
            _k3_call(
                "get_weather",
                _k3_arg("city", "string", "Paris") + _k3_arg("zzz", "string", "x"),
            )
        ),
        # required schema but no argument tags
        _k3_response() + _k3_tools(_k3_call("get_weather", "")),
        # missing tools close marker
        _k3_response()
        + _K3_TOOLS_OPEN
        + _k3_call("get_weather", _k3_arg("city", "string", "Paris")),
        # required but no tool call
        _k3_response("hello"),
    ],
)
def test_kimi_k3_required_rejects_invalid(body: str):
    assert not _is_grammar_accept_string(_k3_grammar("required"), body)


def test_kimi_k3_schema_without_required_accepts_empty_call():
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        )
    ]
    grammar = _k3_grammar("required", tools=tools)
    body = _k3_response() + _k3_tools(_k3_call("get_weather", ""))

    assert _is_grammar_accept_string(grammar, body)


def test_kimi_k3_auto_strict_allows_response_only(sample_tools_strict):
    # With a strict tool the tag is built; the tools channel is optional so a
    # plain response (no tool call) is still valid.
    grammar = _k3_grammar("auto", tools=sample_tools_strict)
    assert _is_grammar_accept_string(grammar, _k3_response("Just answering."))


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_get_model_structural_tag_supports_named_tool_choice(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice=ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name="get_weather")
        ),
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)


@pytest.mark.parametrize(
    ("parser_cls", "model"),
    [
        (DeepSeekV3ToolParser, "deepseek_r1"),
        (DeepSeekV31ToolParser, "deepseek_v3_1"),
        (DeepSeekV32EngineToolParser, "deepseek_v3_2"),
        (DeepSeekV4EngineToolParser, "deepseek_v4"),
        (Glm47MoeModelToolParser, "glm_4_7"),
        (Hermes2ProToolParser, "hermes"),
        (KimiK2ToolParser, "kimi"),
        (Llama3JsonToolParser, "llama"),
        (MinimaxM2ToolParser, "minimax"),
        (Qwen3EngineToolParser, "qwen_3_coder"),
    ],
)
def test_tool_parsers_declare_matching_xgrammar_builtin_model(parser_cls, model):
    assert parser_cls.structural_tag_model == model
    assert not parser_cls.supports_required_and_named


def test_tool_parsers_without_structural_tag_support_required_and_named():
    class NonStructuralTagToolParser(ToolParser):
        pass

    assert NonStructuralTagToolParser.structural_tag_model is None
    assert NonStructuralTagToolParser.supports_required_and_named


def test_non_structural_tag_parser_uses_schema_constraints(
    sample_tools: list[ChatCompletionToolsParam],
):
    parser = ToolParser(MagicMock())
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
    )

    out = parser.adjust_request(request)

    assert out.structured_outputs is not None
    assert out.structured_outputs.json is not None
    assert out.structured_outputs.structural_tag is None


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
    ids=["required", "named"],
)
def test_parser_without_required_and_named_support_skips_schema_constraints(
    tool_choice: str | dict[str, object],
    sample_tools: list[ChatCompletionToolsParam],
):
    """``supports_required_and_named=False`` parsers must not get the tool
    JSON schema installed as a decoding constraint.

    Their models emit a native, non-JSON tool-call syntax (GLM's XML tags,
    thinking special tokens), which the JSON grammar rejects: the FSM fails
    to advance and the request 500s or hangs forever when ``max_tokens`` is
    unset. The serving layer already falls back to auto parsing here, so the
    request must stay unconstrained.
    """
    parser = Glm47MoeModelToolParser(MagicMock(), tools=sample_tools)
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [],
            "model": "m",
            "tools": [tool.model_dump() for tool in sample_tools],
            "tool_choice": tool_choice,
        }
    )

    out = parser.adjust_request(request)

    assert out.structured_outputs is None


def test_get_structural_tag_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    captured: list[bool] = []

    def fake_get_model_structural_tag(*, reasoning: bool, **kwargs):
        captured.append(reasoning)
        return None

    monkeypatch.setattr(
        "vllm.tool_parsers.structural_tag_registry.get_model_structural_tag",
        fake_get_model_structural_tag,
    )

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
    )
    parser = Qwen3EngineToolParser(MagicMock(), tools=sample_tools_strict)

    parser.get_structural_tag(request)

    assert captured == [False]


def test_unified_parser_get_structural_tag_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    captured: list[bool] = []

    def fake_get_model_structural_tag(*, reasoning: bool, **kwargs):
        captured.append(reasoning)
        return None

    monkeypatch.setattr(
        "vllm.tool_parsers.structural_tag_registry.get_model_structural_tag",
        fake_get_model_structural_tag,
    )

    class TestParser(DelegatingParser):
        tool_parser_cls = Qwen3EngineToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
    )
    parser = TestParser(MagicMock(), tools=sample_tools_strict)
    parser.reasoning_parser = MagicMock(adjust_request=lambda request: request)

    parser.adjust_request(request)

    assert captured == [False]


def test_xgrammar_function_parameters_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    captured: list[list[dict]] = []

    def fake_get_xgrammar_model_structural_tag(*, tools: list[dict], **kwargs):
        captured.append(tools)
        return None

    monkeypatch.setattr(
        "vllm.tool_parsers.structural_tag_registry.get_xgrammar_model_structural_tag",
        fake_get_xgrammar_model_structural_tag,
    )

    get_model_structural_tag(
        model="llama",
        tools=sample_tools_strict,
        tool_choice="auto",
        reasoning=False,
    )

    assert (
        captured[0][0]["function"]["parameters"]
        == sample_tools_strict[0].function.parameters
    )
    assert sample_tools_strict[0].function.parameters is not None


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_auto_tool_choice_skips_structural_tag_without_strict(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice="auto",
        reasoning=False,
    )

    assert tag is None


def test_get_function_parameters_relaxes_function_strict_false():
    function = SimpleNamespace(
        parameters={"type": "object", "properties": {}},
        strict=False,
    )

    assert get_function_parameters(function) is True


def _k3_tools_with_root_defs() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "make_config",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "object",
                            "properties": {
                                "build": {"$ref": "#/$defs/build"},
                                "index": {"type": "string"},
                            },
                            "required": ["index"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["config"],
                    "$defs": {
                        "build": {
                            "type": "object",
                            "properties": {"outDir": {"type": "string"}},
                            "additionalProperties": False,
                        }
                    },
                },
            },
        )
    ]


def test_kimi_k3_property_ref_to_root_defs_compiles_and_accepts():
    # Root-level $defs referenced from inside a property schema (the walle
    # TestReferences shape). Slicing the property out of the parameters
    # document orphans "#/$defs/..." unless the builder re-attaches $defs;
    # before the fix Grammar.from_structural_tag raised on the dangling ref.
    grammar = _k3_grammar("required", tools=_k3_tools_with_root_defs())

    body = _k3_response() + _k3_tools(
        _k3_call(
            "make_config",
            _k3_arg(
                "config",
                "object",
                '{"build": {"outDir": "dist"}, "index": "a.html"}',
            ),
        )
    )
    assert _is_grammar_accept_string(grammar, body)


def _k3_tools_with_string_enum() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "set_unit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", " fahrenheit", "\tkelvin"],
                        },
                    },
                    "required": ["unit"],
                },
            },
        )
    ]


@pytest.mark.parametrize("value", ["celsius", " fahrenheit", "\tkelvin"])
def test_kimi_k3_string_enum_accepts_exact_values(value: str):
    # Raw string channel with enum: constrained to the exact enum values,
    # including leading-whitespace variants the model otherwise flubs.
    grammar = _k3_grammar("required", tools=_k3_tools_with_string_enum())
    body = _k3_response() + _k3_tools(
        _k3_call("set_unit", _k3_arg("unit", "string", value))
    )
    assert _is_grammar_accept_string(grammar, body)


@pytest.mark.parametrize("value", ["kelvin", "Celsius", "celsius ", ""])
def test_kimi_k3_string_enum_rejects_non_members(value: str):
    grammar = _k3_grammar("required", tools=_k3_tools_with_string_enum())
    body = _k3_response() + _k3_tools(
        _k3_call("set_unit", _k3_arg("unit", "string", value))
    )
    assert not _is_grammar_accept_string(grammar, body)


def _k3_tools_with_maxlen() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "set_note",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {"type": "string", "maxLength": 8, "minLength": 2},
                    },
                    "required": ["note"],
                },
            },
        )
    ]


def test_kimi_k3_string_maxlength_bounds_raw_channel():
    # Raw string channel with maxLength/minLength: enforced via a bounded
    # regex that keeps the "<|" marker prefix unambiguous but still allows
    # a bare '<' inside values.
    grammar = _k3_grammar("required", tools=_k3_tools_with_maxlen())

    def body(val: str) -> str:
        return _k3_response() + _k3_tools(
            _k3_call("set_note", _k3_arg("note", "string", val))
        )

    assert _is_grammar_accept_string(grammar, body("ab"))
    assert _is_grammar_accept_string(grammar, body("a<b then"))
    assert not _is_grammar_accept_string(grammar, body("way too long note"))
    assert not _is_grammar_accept_string(grammar, body("a"))  # under minLength


def test_kimi_k3_forced_tool_choice_builds_single_mandatory_call():
    # Named tool choice normalizes to "forced": the tag must require exactly
    # the named tool's call (no response-only escape).
    grammar = _k3_grammar(
        ChatCompletionNamedToolChoiceParam(
            type="function",
            function=ChatCompletionNamedFunction(name="get_weather"),
        ),
        tools=_k3_tools_by_name(),
    )

    ok = _k3_response() + _k3_tools(
        _k3_call("get_weather", _k3_arg("city", "string", "Paris"))
    )
    response_only = _k3_response("no call here")
    assert _is_grammar_accept_string(grammar, ok)
    assert not _is_grammar_accept_string(grammar, response_only)
