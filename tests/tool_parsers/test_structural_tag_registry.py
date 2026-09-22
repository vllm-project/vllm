# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from vllm.tool_parsers.deepseekv41_engine_tool_parser import DeepSeekV41EngineToolParser
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
    ToolChoice,
    get_function_parameters,
    get_model_structural_tag,
)
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel


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


def test_deepseek_v41_named_choice_emits_one_call(sample_tools):
    tag = get_model_structural_tag(
        "deepseek_v4_1",
        sample_tools,
        ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name="get_weather")
        ),
        reasoning=False,
    )
    grammar = Grammar.from_structural_tag(tag)
    call = (
        '<｜DSML｜ invoke name="get_weather">\n'
        '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
    )
    assert _is_grammar_accept_string(
        grammar, "\n\n<｜DSML｜ calls>\n" + call + "</｜DSML｜ calls>"
    )
    assert not _is_grammar_accept_string(
        grammar, "\n\n<｜DSML｜ calls>\n" + call * 2 + "</｜DSML｜ calls>"
    )
    assert (
        get_model_structural_tag("deepseek_v4_1", sample_tools, "auto", reasoning=False)
        is None
    )


@pytest.mark.parametrize("choice", ["auto", "required", "get_weather"])
def test_deepseek_v41_constrains_parameters_after_reasoning(
    sample_tools_strict, choice
):
    """Strict calls must enforce the schema without requiring another think close."""
    tool_choice = (
        ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name=choice)
        )
        if choice == "get_weather"
        else choice
    )
    sample_tools_strict[0].function.parameters["additionalProperties"] = False
    tag = get_model_structural_tag(
        "deepseek_v4_1", sample_tools_strict, tool_choice, reasoning=False
    )
    grammar = Grammar.from_structural_tag(tag)
    begin = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    end = "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    parameter = (
        '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
    )
    assert _is_grammar_accept_string(grammar, begin + parameter + end)
    for invalid in (
        "",  # missing required city
        parameter * 2,
        parameter.replace('name="city"', 'name="unknown"'),
        parameter.replace('string="true">Paris', 'string="false">42'),
        parameter.replace("｜ parameter", "｜parameter"),
        '{"city":"Paris"}',  # the encoder emits DSML parameters, not a JSON body
    ):
        assert not _is_grammar_accept_string(grammar, begin + invalid + end)
    assert not _is_grammar_accept_string(
        grammar, "reason</think>" + begin + parameter + end
    )
    assert _is_grammar_accept_string(grammar, "Hello") == (choice == "auto")


def test_deepseek_v41_non_strict_parallel_calls_keep_typed_dsml(sample_tools):
    sample_tools[0].function.strict = False
    tag = get_model_structural_tag(
        "deepseek_v4_1", sample_tools, "required", reasoning=False
    )
    grammar = Grammar.from_structural_tag(tag)
    call = (
        '<｜DSML｜ invoke name="get_weather">\n'
        '<｜DSML｜ parameter name="extra" string="false">'
        '{"nested":[true,null,1.5]}</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
    )
    output = "\n\n<｜DSML｜ calls>\n" + call * 2 + "</｜DSML｜ calls>"
    assert _is_grammar_accept_string(grammar, output)
    assert not _is_grammar_accept_string(
        grammar, output.replace('{"nested":[true,null,1.5]}', "invalid")
    )


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
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model="hermes",
        tools=sample_tools_strict,
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


def test_hermes_required_without_strict_leaves_arguments_free(
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model="hermes",
        tools=sample_tools,
        tool_choice="required",
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)
    tags = tag.model_dump()["format"]["tags"]
    assert tags
    assert all(tag_dump["content"]["json_schema"] is True for tag_dump in tags)


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
_K3_END_OF_MSG = "<|end_of_msg|>"


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


def _as_strict(tools: list[ChatCompletionToolsParam]) -> list[ChatCompletionToolsParam]:
    for tool in tools:
        tool.function.strict = True
    return tools


def _k3_grammar(tool_choice, tools=None, *, strict: bool = True):
    """Build the K3 grammar; tools are strict by default so the typed
    argument channel is exercised (an unset ``strict`` leaves arguments free)."""
    tools = tools if tools is not None else _k3_tools_by_name()
    tag = get_model_structural_tag(
        model="kimi_k3",
        tools=_as_strict(tools) if strict else tools,
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


def test_kimi_k3_required_without_strict_leaves_arguments_free():
    # An unset ``strict`` pins the call envelope only: a value the schema
    # would reject is accepted, and the declared-required argument may be
    # omitted.
    grammar = _k3_grammar("required", strict=False)
    body = _k3_response() + _k3_tools(
        _k3_call("get_weather", _k3_arg("days", "number", "abc"))
    )
    assert _is_grammar_accept_string(grammar, body)
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


@pytest.mark.parametrize(
    "body",
    [
        _K3_RESPONSE_OPEN + "answer<|close|>message",
        _K3_RESPONSE_OPEN + "answer<|open|>tools",
        _K3_RESPONSE_OPEN + "answer" + _K3_END_OF_MSG,
    ],
)
def test_kimi_k3_response_body_rejects_reserved_marker_prefixes(body: str):
    assert not _is_grammar_accept_string(
        _k3_grammar("required"), body, require_termination=False
    )


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
        (DeepSeekV41EngineToolParser, "deepseek_v4_1"),
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


@pytest.mark.parametrize(
    "parser_cls", [Qwen3EngineToolParser, DeepSeekV41EngineToolParser]
)
def test_unified_parser_get_structural_tag_disables_reasoning(
    parser_cls,
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
        tool_parser_cls = parser_cls

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
    )
    parser = TestParser(MagicMock(), tools=sample_tools_strict)
    parser._reasoning_parser = MagicMock(adjust_request=lambda request: request)

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


# ===========================================================================
# Note(arpera): testing corner case tool_choice="auto" + response_format:
#
# Firstly, we convert both tools and response_format to structural tags.
# Then we combine them using OR node resulting in one structural tag.
# See https://github.com/vllm-project/vllm/issues/39929 and the PR #56086
# review discussion for context.
# ===========================================================================


def _json_schema_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }


def test_apply_structural_tag_ors_json_schema_response_format(
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    class TestParser(DelegatingParser):
        tool_parser_cls = Qwen3EngineToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
        response_format=_json_schema_response_format(),
    )
    parser = TestParser(MagicMock(), tools=sample_tools_strict)

    out = parser.adjust_request(request)

    assert out.response_format is None
    assert out.structured_outputs is not None
    assert out.structured_outputs.structural_tag is not None
    tag = json.loads(out.structured_outputs.structural_tag)
    assert tag["type"] == "structural_tag"
    assert tag["format"]["type"] == "or"
    assert len(tag["format"]["elements"]) == 2

    # The combined grammar accepts response_format-compliant text with no
    # tool call, not just a tool call.
    grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
    assert _is_grammar_accept_string(grammar, '{"text": "hi"}')


def test_apply_structural_tag_falls_back_when_model_lacks_support(
    sample_tools: list[ChatCompletionToolsParam],
):
    # sample_tools has no "strict": True, so get_model_structural_tag()
    # returns None for tool_choice="auto" (structural_tag_registry.py's
    # "auto and not any_tool_strict" guard) — main's fallback behavior
    # applies: response_format wins, tool calls are not constrained.
    class TestParser(DelegatingParser):
        tool_parser_cls = Qwen3EngineToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="auto",
        response_format=_json_schema_response_format(),
    )
    parser = TestParser(MagicMock(), tools=sample_tools)

    with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
        out = parser.adjust_request(request)

    assert out.response_format is not None
    assert out.structured_outputs is None
    mock_warn.assert_called_once()


def test_apply_structural_tag_drops_unsupported_response_format_type_with_warning(
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    # json_object has no schema to OR in, so it's out of scope for now:
    # tool calling takes priority (matches the pre-OR default), with a
    # warning instead of a silent drop.
    class TestParser(DelegatingParser):
        tool_parser_cls = Qwen3EngineToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    parser = TestParser(MagicMock(), tools=sample_tools_strict)

    with patch("vllm.parser.abstract_parser.logger.warning_once") as mock_warn:
        out = parser.adjust_request(request)

    assert out.response_format is None
    assert out.structured_outputs is not None
    tag = json.loads(out.structured_outputs.structural_tag)
    assert tag["format"]["type"] != "or"
    mock_warn.assert_called_once()


def _pins_argument_schema(tag: StructuralTag) -> bool:
    return '"json_schema": {' in json.dumps(tag.model_dump(), ensure_ascii=False)


def _dumped(tag: StructuralTag) -> str:
    return json.dumps(tag.model_dump(), ensure_ascii=False)


def test_tool_strict_level_auto_is_the_default(
    sample_tools: list[ChatCompletionToolsParam],
):
    """Auto + no strict tool gets no tag unless the operator raises the floor."""
    assert (
        get_model_structural_tag(
            model="deepseek_v4",
            tools=sample_tools,
            tool_choice="auto",
            reasoning=False,
        )
        is None
    )


@pytest.mark.parametrize("model", sorted(SUPPORTED_STRUCTURAL_TAG_MODELS))
def test_tool_strict_level_function_lifts_auto_gate(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice="auto",
        reasoning=False,
        strict_level=ToolStrictLevel.FUNCTION,
    )

    assert tag is not None


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_tool_strict_level_function_pins_envelope_only(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
):
    """The request's own tools stay untouched; unset ``strict`` stays free."""
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice="auto",
        reasoning=False,
        strict_level=ToolStrictLevel.FUNCTION,
    )

    assert tag is not None
    assert not _pins_argument_schema(tag)
    assert all(tool.function.strict is None for tool in sample_tools)


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_tool_strict_level_parameter_pins_argument_schemas(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice="auto",
        reasoning=False,
        strict_level=ToolStrictLevel.PARAMETER,
    )

    assert tag is not None
    assert _pins_argument_schema(tag)


def test_tool_strict_level_function_keeps_client_strict_tools_strict(
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model="deepseek_v4",
        tools=sample_tools_strict,
        tool_choice="auto",
        reasoning=False,
        strict_level=ToolStrictLevel.FUNCTION,
    )

    assert tag is not None
    assert _pins_argument_schema(tag)


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_absent_strict_is_not_pinned_next_to_a_strict_tool(model: str):
    """A strict tool must not drag a neighbour with unset ``strict`` into
    schema enforcement."""
    weather = ChatCompletionToolsParam(
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
    search = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    )
    tag = get_model_structural_tag(
        model=model,
        tools=[weather, search],
        tool_choice="auto",
        reasoning=False,
    )

    assert tag is not None
    dumped = _dumped(tag)
    assert '"city"' in dumped
    assert '"query"' not in dumped


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name="get_weather")
        ),
    ],
)
@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_forced_tool_choice_pins_schema_only_for_strict_tools(
    model: str,
    tool_choice: ToolChoice,
    sample_tools: list[ChatCompletionToolsParam],
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    """Required / named always constrain the call; the argument schema
    follows the per-tool ``strict`` (or the parameter level)."""
    free = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice=tool_choice,
        reasoning=False,
        strict_level=ToolStrictLevel.FUNCTION,
    )
    pinned = get_model_structural_tag(
        model=model,
        tools=sample_tools_strict,
        tool_choice=tool_choice,
        reasoning=False,
    )
    floored = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice=tool_choice,
        reasoning=False,
        strict_level=ToolStrictLevel.PARAMETER,
    )

    assert free is not None and not _pins_argument_schema(free)
    assert pinned is not None and _pins_argument_schema(pinned)
    assert floored is not None and _pins_argument_schema(floored)


def test_tool_strict_level_parameter_overrides_client_strict_false():
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            },
        )
    ]
    tag = get_model_structural_tag(
        model="deepseek_v4",
        tools=tools,
        tool_choice="required",
        reasoning=False,
        strict_level=ToolStrictLevel.PARAMETER,
    )

    assert tag is not None
    assert _pins_argument_schema(tag)


def test_tool_strict_level_from_name():
    assert ToolStrictLevel.from_name("AUTO") is ToolStrictLevel.AUTO
    assert ToolStrictLevel.from_name("Parameter") is ToolStrictLevel.PARAMETER
    with pytest.raises(ValueError, match="expected one of auto, function, parameter"):
        ToolStrictLevel.from_name("strict")
    with pytest.raises(ValueError, match="expected one of auto, function, parameter"):
        ToolStrictLevel.from_name("off")
