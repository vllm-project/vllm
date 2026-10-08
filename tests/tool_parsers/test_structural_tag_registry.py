# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types.responses import ToolChoiceFunction
from openai.types.responses.tool_choice_allowed import ToolChoiceAllowed
from xgrammar import Grammar, StructuralTag
from xgrammar import get_model_structural_tag as get_xgrammar_builtin_structural_tag
from xgrammar.testing import _is_grammar_accept_string

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_tool_dicts
from vllm.exceptions import VLLMValidationError
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.plamo3 import (
    BEGIN_TOOL_ARGUMENTS,
    BEGIN_TOOL_NAME,
    BEGIN_TOOL_REQUEST,
    BEGIN_TOOL_REQUESTS,
    END_TOOL_ARGUMENTS,
    END_TOOL_NAME,
    END_TOOL_REQUEST,
    END_TOOL_REQUESTS,
    EOT,
)
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
from vllm.tool_parsers.mimo_tool_parser import MiMoToolParser
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser
from vllm.tool_parsers.plamo3_engine_tool_parser import Plamo3EngineToolParser
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.step3p5_tool_parser import Step3p5ToolParser
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


_MCP_TOOL = {"type": "mcp", "server_label": "docs", "server_url": "http://mcp"}


def _namespace_tool(strict: bool | None = None) -> dict[str, Any]:
    lookup = {
        "type": "function",
        "name": "lookup",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
    }
    if strict is not None:
        lookup["strict"] = strict
    return {"type": "namespace", "name": "crm", "description": "", "tools": [lookup]}


def _hermes_call(name: str) -> str:
    return (
        f'<tool_call>\n{{"name": "{name}", "arguments": {{"id": "1"}}}}\n</tool_call>'
    )


@pytest.mark.parametrize("model", sorted(SUPPORTED_STRUCTURAL_TAG_MODELS))
def test_no_structural_tag_without_function_tools(model: str):
    """Builtin tools never reach the prompt, so there is nothing to constrain."""
    request = ResponsesRequest.model_validate(
        {"input": "hi", "tools": [_MCP_TOOL, {"type": "web_search"}]}
    )

    tag = get_model_structural_tag(
        model=model,
        tools=request.tools,
        tool_choice="auto",
        reasoning=False,
        strict_level=ToolStrictLevel.FUNCTION,
    )

    assert tag is None
    with pytest.raises(VLLMValidationError, match="no function tool"):
        get_model_structural_tag(
            model=model,
            tools=request.tools,
            tool_choice="required",
            reasoning=False,
        )


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_structural_tag_allows_exactly_the_prompt_tools(tool_choice: str):
    """The grammar lets the model call every tool its prompt lists, and no other."""
    request = ResponsesRequest.model_validate(
        {
            "input": "hi",
            "tools": [
                {"type": "function", "name": "get_weather", "parameters": {}},
                _namespace_tool(),
                _MCP_TOOL,
            ],
        }
    )
    prompt_names = [
        tool["function"]["name"]
        for tool in construct_tool_dicts(request.tools, tool_choice)
    ]

    tag = get_model_structural_tag(
        model="hermes",
        tools=request.tools,
        tool_choice=tool_choice,
        reasoning=False,
        strict_level=ToolStrictLevel.FUNCTION,
    )

    assert prompt_names == ["get_weather", "crm__lookup"]
    assert tag is not None
    grammar = Grammar.from_structural_tag(json.dumps(tag.model_dump()))
    for name in prompt_names:
        assert _is_grammar_accept_string(grammar, _hermes_call(name))
    assert not _is_grammar_accept_string(grammar, _hermes_call("docs"))


def test_strict_namespace_function_enables_auto_structural_tag():
    request = ResponsesRequest.model_validate(
        {"input": "hi", "tools": [_namespace_tool(strict=True)]}
    )

    tag = get_model_structural_tag(
        model="hermes", tools=request.tools, tool_choice="auto", reasoning=False
    )

    assert tag is not None
    grammar = Grammar.from_structural_tag(json.dumps(tag.model_dump()))
    assert _is_grammar_accept_string(grammar, _hermes_call("crm__lookup"))


def test_harmony_structural_tag_keeps_builtin_tools():
    """Harmony can call builtin tools, so they still shape its grammar."""
    request = ResponsesRequest.model_validate(
        {"input": "hi", "tools": [{"type": "web_search"}]}
    )

    tag = get_model_structural_tag(
        model="harmony",
        tools=request.tools,
        tool_choice="required",
        reasoning=False,
    )

    assert tag is not None


def test_glm47_default_level_allows_builtin_only_tools(
    monkeypatch: pytest.MonkeyPatch,
):
    """GLM-4.7's default FUNCTION level must not reject builtin-only tools."""
    monkeypatch.setattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", True)

    class TestParser(DelegatingParser):
        tool_parser_cls = Glm47MoeModelToolParser

    request = ResponsesRequest.model_validate({"input": "hi", "tools": [_MCP_TOOL]})
    parser = TestParser(MagicMock(), tools=None)
    parser._reasoning_parser = MagicMock(adjust_request=lambda request: request)

    assert parser.adjust_request(request).structured_outputs is None


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


def _plamo3_call(name: str, arguments: str) -> str:
    return (
        BEGIN_TOOL_REQUEST
        + BEGIN_TOOL_NAME
        + name
        + END_TOOL_NAME
        + BEGIN_TOOL_ARGUMENTS
        + arguments
        + END_TOOL_ARGUMENTS
        + END_TOOL_REQUEST
    )


def _plamo3_requests(*calls: str, eot: bool = False) -> str:
    return (
        BEGIN_TOOL_REQUESTS + "".join(calls) + END_TOOL_REQUESTS + (EOT if eot else "")
    )


def _plamo3_grammar(tool_choice, tools):
    tag = get_model_structural_tag(
        model="plamo3",
        tools=tools,
        tool_choice=tool_choice,
        reasoning=False,
    )
    assert isinstance(tag, StructuralTag)
    return Grammar.from_structural_tag(tag)


def _plamo3_tools() -> list[ChatCompletionToolsParam]:
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
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_time",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                },
            },
        ),
    ]


def test_plamo3_registered_as_vllm_structural_tag_model():
    assert "plamo3" in VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
    assert Plamo3EngineToolParser.structural_tag_model == "plamo3"
    assert Plamo3EngineToolParser.supports_required_and_named is False


@pytest.mark.parametrize(
    "body",
    [
        _plamo3_requests(_plamo3_call("get_weather", '{"city":"Tokyo"}')),
        _plamo3_requests(
            _plamo3_call("get_weather", '{"city":"Tokyo"}'),
            _plamo3_call("get_time", '{"timezone":"Asia/Tokyo"}'),
            eot=True,
        ),
    ],
)
def test_plamo3_required_accepts_complete_request_blocks(body: str):
    assert _is_grammar_accept_string(_plamo3_grammar("required", _plamo3_tools()), body)


@pytest.mark.parametrize(
    "body",
    [
        _plamo3_requests(_plamo3_call("unknown", '{"city":"Tokyo"}')),
        _plamo3_requests(_plamo3_call("get_weather", "{}")),
        BEGIN_TOOL_REQUESTS + _plamo3_call("get_weather", '{"city":"Tokyo"}'),
    ],
)
def test_plamo3_required_rejects_invalid_request_blocks(body: str):
    assert not _is_grammar_accept_string(
        _plamo3_grammar("required", _plamo3_tools()),
        body,
    )


def test_plamo3_auto_allows_text_or_a_tool_request_block(sample_tools_strict):
    grammar = _plamo3_grammar("auto", sample_tools_strict)

    assert _is_grammar_accept_string(grammar, "Plain response")
    assert _is_grammar_accept_string(
        grammar,
        _plamo3_requests(_plamo3_call("get_weather", '{"city":"Tokyo"}')),
    )


def test_plamo3_forced_stops_after_the_named_tool_call():
    tools = _plamo3_tools()
    tool_choice = ChatCompletionNamedToolChoiceParam(
        function=ChatCompletionNamedFunction(name="get_weather")
    )
    grammar = _plamo3_grammar(tool_choice, tools)

    assert _is_grammar_accept_string(
        grammar,
        _plamo3_requests(_plamo3_call("get_weather", '{"city":"Tokyo"}')),
    )
    assert not _is_grammar_accept_string(
        grammar,
        _plamo3_requests(
            _plamo3_call("get_weather", '{"city":"Tokyo"}'),
            _plamo3_call("get_time", '{"timezone":"Asia/Tokyo"}'),
        ),
    )


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
        (Step3p5ToolParser, "qwen_3_coder"),
        (MiMoToolParser, "mimo"),
    ],
)
def test_tool_parsers_declare_matching_xgrammar_builtin_model(parser_cls, model):
    assert parser_cls.structural_tag_model == model
    assert not parser_cls.supports_required_and_named


@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get_weather"}}],
)
def test_step3p5_forced_tool_choice_round_trips_native_xml(
    sample_tools: list[ChatCompletionToolsParam], tool_choice
):
    """Forced Step calls must be constrained to, and parsed from, Step XML."""

    class StepParser(DelegatingParser):
        tool_parser_cls = Step3p5ToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        # Named tool choice validation reads tools in their JSON form.
        tools=[tool.model_dump(exclude_none=True) for tool in sample_tools],
        tool_choice=tool_choice,
    )
    parser = StepParser(MagicMock(), tools=sample_tools)
    request = parser.adjust_request(request)

    assert request.structured_outputs.json is None
    grammar = Grammar.from_structural_tag(request.structured_outputs.structural_tag)
    # Rendered exactly as the Step-3.5/3.7 chat templates render tool calls.
    output = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nDallas\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert _is_grammar_accept_string(grammar, output)
    assert not _is_grammar_accept_string(grammar, "It is sunny in Dallas.")

    _, content, tool_calls = parser.parse(output, request, enable_auto_tools=True)

    assert not content
    assert [(call.name, json.loads(call.arguments)) for call in tool_calls] == [
        ("get_weather", {"city": "Dallas"})
    ]


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


_NAMED_CHAT_TOOL_CHOICE = {"type": "function", "function": {"name": "get_weather"}}


def _responses_request(tool_choice) -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": tool_choice,
        }
    )


@pytest.mark.parametrize("tool_choice", ["required", _NAMED_CHAT_TOOL_CHOICE])
def test_native_format_parser_skips_json_schema_constraint(
    sample_tools: list[ChatCompletionToolsParam],
    tool_choice,
):
    # Parsers that extract required/named calls from their native format must
    # not be forced into JSON, which their extractor cannot parse.
    class NativeFormatToolParser(ToolParser):
        supports_required_and_named = False

    parser = NativeFormatToolParser(MagicMock())
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        # Named tool choice validation reads tools in their JSON form.
        tools=[tool.model_dump(exclude_none=True) for tool in sample_tools],
        tool_choice=tool_choice,
    )

    out = parser.adjust_request(request)

    assert out.structured_outputs is None


@pytest.mark.parametrize(
    "tool_choice",
    ["required", ToolChoiceFunction(type="function", name="get_weather")],
)
def test_native_format_parser_skips_json_schema_constraint_responses(
    tool_choice,
):
    class NativeFormatToolParser(ToolParser):
        supports_required_and_named = False

    parser = NativeFormatToolParser(MagicMock())
    request = _responses_request(tool_choice)

    out = parser.adjust_request(request)

    assert out.structured_outputs is None
    assert out.text is None


@pytest.mark.parametrize("tool_choice", ["required", _NAMED_CHAT_TOOL_CHOICE])
def test_glm47_without_strict_tool_calling_skips_json_schema_constraint(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools: list[ChatCompletionToolsParam],
    tool_choice,
):
    # With strict tool calling off, no structural tag is attached, so GLM-4.7
    # must decode its native XML instead of a JSON tool-call list.
    monkeypatch.setattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", False)

    class TestParser(DelegatingParser):
        tool_parser_cls = Glm47MoeModelToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        # Named tool choice validation reads tools in their JSON form.
        tools=[tool.model_dump(exclude_none=True) for tool in sample_tools],
        tool_choice=tool_choice,
    )
    parser = TestParser(MagicMock(), tools=sample_tools)
    parser._reasoning_parser = MagicMock(adjust_request=lambda request: request)

    out = parser.adjust_request(request)

    assert out.structured_outputs is None
    assert out.skip_special_tokens is False


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
    "parser_cls", [Qwen3EngineToolParser, DeepSeekV41EngineToolParser, MiMoToolParser]
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


# ---------------------------------------------------------------------------
# GLM-4.7 structural tag: builtin content for strict tools, shallow otherwise
# ---------------------------------------------------------------------------


def _glm47_tools() -> list[ChatCompletionToolsParam]:
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
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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
                },
            },
        ),
    ]


def _glm47_arg(key: str, value: str) -> str:
    return f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"


def _glm47_call(name: str, *args: str) -> str:
    return f"<tool_call>{name}{''.join(args)}</tool_call>"


def _glm47_shallow_grammar(
    tool_choice, tools=None, reasoning: bool = False, *, auto: bool = False
):
    # auto needs the FUNCTION level to lift the non-strict auto gate.
    tag = get_model_structural_tag(
        model="glm_4_7",
        tools=tools if tools is not None else _glm47_tools(),
        tool_choice=tool_choice,
        reasoning=reasoning,
        strict_level=ToolStrictLevel.FUNCTION if auto else ToolStrictLevel.AUTO,
    )
    assert isinstance(tag, StructuralTag)
    return Grammar.from_structural_tag(tag)


@pytest.mark.parametrize(
    "body",
    [
        # plain text, no tool call
        "Hello there",
        # text followed by a call
        "Sure." + _glm47_call("get_weather", _glm47_arg("city", "Paris")),
        # call only, required arg present
        _glm47_call("get_weather", _glm47_arg("city", "Paris")),
        # all arguments omitted
        _glm47_call("get_weather"),
        # repeated argument
        _glm47_call(
            "get_weather", _glm47_arg("city", "Paris"), _glm47_arg("city", "Rome")
        ),
        # arguments in reverse declaration order
        _glm47_call(
            "get_weather", _glm47_arg("days", "3"), _glm47_arg("city", "Paris")
        ),
        # enum-constrained value
        _glm47_call("get_weather", _glm47_arg("unit", "celsius")),
        # string values with spaces and a bare '<'
        _glm47_call("run_command", _glm47_arg("command", "grep 'a<b' x.py")),
        # second tool, two calls
        _glm47_call("run_command", _glm47_arg("command", "ls -la"))
        + _glm47_call("get_weather", _glm47_arg("city", "Paris")),
    ],
)
def test_glm47_non_strict_auto_accepts_valid_tool_calls(body: str):
    grammar = _glm47_shallow_grammar("auto", auto=True)
    assert _is_grammar_accept_string(grammar, body)


@pytest.mark.parametrize(
    "body",
    [
        # unknown tool name
        _glm47_call("unknown_tool", _glm47_arg("city", "Paris")),
        # undeclared argument key
        _glm47_call("get_weather", _glm47_arg("zzz", "x")),
        # structural marker inside a string value
        _glm47_call("get_weather", _glm47_arg("city", "a<arg_key>b")),
        _glm47_call("get_weather", _glm47_arg("city", "a</tool_call>b")),
        # integer-typed value given non-numeric text
        _glm47_call("get_weather", _glm47_arg("days", "abc")),
        # enum-typed value outside the enum
        _glm47_call("get_weather", _glm47_arg("unit", "kelvin")),
        # unterminated tool call
        "<tool_call>get_weather" + _glm47_arg("city", "Paris"),
    ],
)
def test_glm47_non_strict_auto_rejects_invalid(body: str):
    grammar = _glm47_shallow_grammar("auto", auto=True)
    assert not _is_grammar_accept_string(grammar, body)


def test_glm47_non_strict_required_requires_at_least_one_call():
    grammar = _glm47_shallow_grammar("required")

    assert not _is_grammar_accept_string(grammar, "Just answering.")
    assert _is_grammar_accept_string(
        grammar, _glm47_call("get_weather", _glm47_arg("city", "Paris"))
    )
    assert _is_grammar_accept_string(
        grammar,
        _glm47_call("run_command", _glm47_arg("command", "ls"))
        + _glm47_call("get_weather", _glm47_arg("city", "Paris")),
    )


def test_glm47_non_strict_forced_emits_single_named_call():
    grammar = _glm47_shallow_grammar(
        ChatCompletionNamedToolChoiceParam(
            type="function",
            function=ChatCompletionNamedFunction(name="get_weather"),
        )
    )

    assert _is_grammar_accept_string(
        grammar, _glm47_call("get_weather", _glm47_arg("city", "Paris"))
    )
    assert not _is_grammar_accept_string(grammar, "No call here")
    assert not _is_grammar_accept_string(grammar, _glm47_call("run_command"))
    assert not _is_grammar_accept_string(
        grammar,
        _glm47_call("get_weather", _glm47_arg("city", "Paris"))
        + _glm47_call("run_command"),
    )


def test_glm47_non_strict_reasoning_gates_on_think_close():
    grammar = _glm47_shallow_grammar("required", reasoning=True)
    call = _glm47_call("get_weather", _glm47_arg("city", "Paris"))

    assert _is_grammar_accept_string(grammar, "thinking...</think>" + call)
    assert not _is_grammar_accept_string(grammar, call)


@pytest.mark.parametrize(
    ("parameters", "key"),
    [
        (
            {
                "$ref": "#/$defs/args",
                "$defs": {
                    "args": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    }
                },
            },
            "city",
        ),
        (
            {
                "anyOf": [
                    {"type": "object", "properties": {"city": {"type": "string"}}},
                    {"type": "null"},
                ]
            },
            "city",
        ),
        (
            {"allOf": [{"type": "object", "properties": {"city": {"type": "string"}}}]},
            "city",
        ),
        ({"type": "object", "patternProperties": {"^arg": {"type": "string"}}}, "city"),
        ({"type": "object"}, "city"),
        ({}, "city"),
        (
            {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "additionalProperties": True,
            },
            "undeclared_key",
        ),
    ],
)
def test_glm47_non_plain_object_schemas_keep_free_arguments(parameters, key):
    # Schemas that are not a plain object with declared properties fall back
    # to the builtin's non-strict envelope content, which accepts any key.
    tools = [
        ChatCompletionToolsParam(
            type="function", function={"name": "f", "parameters": parameters}
        )
    ]
    tag = get_model_structural_tag(
        model="glm_4_7",
        tools=tools,
        tool_choice="required",
        reasoning=False,
    )
    assert isinstance(tag, StructuralTag)
    grammar = Grammar.from_structural_tag(tag)
    call = _glm47_call("f", _glm47_arg(key, "Paris"))
    assert _is_grammar_accept_string(grammar, call)


def _glm47_strict_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "strict": True,
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "run_command",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={"name": "no_params", "strict": True, "parameters": None},
        ),
    ]


@pytest.mark.parametrize(
    ("tool_choice", "reasoning"),
    [
        ("auto", False),
        ("required", False),
        ("named", False),
        ("required", True),
        ("allowed", False),
    ],
)
def test_glm47_strict_only_matches_xgrammar_builtin(tool_choice: str, reasoning: bool):
    # Strict-only requests stay byte-identical to the xgrammar builtin.
    strict_tools = _glm47_strict_tools()
    dumped = [
        {"type": "function", "function": t.function.model_dump(exclude_none=True)}
        for t in strict_tools
    ]
    if tool_choice == "named":
        choice = ChatCompletionNamedToolChoiceParam(
            type="function",
            function=ChatCompletionNamedFunction(name="run_command"),
        )
        builtin_choice: Any = {"type": "function", "function": {"name": "run_command"}}
    elif tool_choice == "allowed":
        choice = ToolChoiceAllowed(
            type="allowed_tools",
            mode="auto",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        builtin_choice = {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        }
    else:
        choice = tool_choice
        builtin_choice = tool_choice

    ours = get_model_structural_tag(
        model="glm_4_7",
        tools=strict_tools,
        tool_choice=choice,
        reasoning=reasoning,
    )
    expected = get_xgrammar_builtin_structural_tag(
        model="glm_4_7",
        tools=dumped,
        tool_choice=builtin_choice,
        reasoning=reasoning,
    )

    assert isinstance(ours, StructuralTag)
    assert ours.model_dump() == expected.model_dump()


def test_glm47_parameter_level_pins_schemas_like_the_builtin(
    sample_tools: list[ChatCompletionToolsParam],
):
    # The operator floor overrides per-tool strictness: every tool is pinned.
    ours = get_model_structural_tag(
        model="glm_4_7",
        tools=sample_tools,
        tool_choice="required",
        reasoning=False,
        strict_level=ToolStrictLevel.PARAMETER,
    )
    expected = get_xgrammar_builtin_structural_tag(
        model="glm_4_7",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "strict": True,
                    "parameters": sample_tools[0].function.parameters,
                },
            }
        ],
        tool_choice="required",
        reasoning=False,
    )

    assert isinstance(ours, StructuralTag)
    assert ours.model_dump() == expected.model_dump()


def test_glm47_mixed_tools_dispatch_per_tool(
    sample_tools_strict: list[ChatCompletionToolsParam],
    sample_tools: list[ChatCompletionToolsParam],
):
    mixed = [sample_tools_strict[0], sample_tools[0]]
    mixed[1] = mixed[1].model_copy(deep=True)
    mixed[1].function.name = "run_command"

    tag = get_model_structural_tag(
        model="glm_4_7",
        tools=mixed,
        tool_choice="required",
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)
    tags = tag.model_dump()["format"]["tags"]
    assert tags[0]["content"]["json_schema"] == (
        sample_tools_strict[0].function.parameters
    )
    assert tags[1]["content"]["type"] != "json_schema"


def _glm47_parser(
    tools, monkeypatch: pytest.MonkeyPatch, *, strict_calling: bool = True
) -> Glm47MoeModelToolParser:
    monkeypatch.setattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", strict_calling)
    return Glm47MoeModelToolParser(MagicMock(), tools=tools)


@pytest.mark.parametrize(
    ("tool_choice", "tools"),
    [("auto", "plain"), ("required", "plain"), ("required", "strict")],
)
def test_glm47_get_structural_tag_disabled_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools: list[ChatCompletionToolsParam],
    sample_tools_strict: list[ChatCompletionToolsParam],
    tool_choice: str,
    tools: str,
):
    parser = _glm47_parser(sample_tools, monkeypatch, strict_calling=False)
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict if tools == "strict" else sample_tools,
        tool_choice=tool_choice,
    )

    assert parser.get_structural_tag(request) is None


def test_glm47_default_strict_level_constrains_auto(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools: list[ChatCompletionToolsParam],
):
    # --tool-strict-level auto: the parser's FUNCTION default lifts the
    # non-strict auto gate.
    monkeypatch.setattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", True)

    class TestParser(DelegatingParser):
        tool_parser_cls = Glm47MoeModelToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=[tool.model_dump(exclude_none=True) for tool in sample_tools],
        tool_choice="auto",
    )
    parser = TestParser(MagicMock(), tools=sample_tools)
    parser._reasoning_parser = MagicMock(adjust_request=lambda request: request)

    out = parser.adjust_request(request)

    assert out.structured_outputs is not None
    assert out.structured_outputs.json is None
    grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
    call = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>"
    )
    assert _is_grammar_accept_string(grammar, "Sure. " + call)
    assert _is_grammar_accept_string(grammar, "Plain answer.")
    assert not _is_grammar_accept_string(
        grammar, "Sure. " + call.replace("city", "ctiy")
    )


def test_glm47_responses_request_attaches_shallow_tag(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", True)

    class TestParser(DelegatingParser):
        tool_parser_cls = Glm47MoeModelToolParser

    request = ResponsesRequest.model_validate(
        {
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": "required",
        }
    )
    parser = TestParser(MagicMock(), tools=None)
    parser._reasoning_parser = MagicMock(adjust_request=lambda request: request)

    out = parser.adjust_request(request)

    assert out.structured_outputs is not None
    assert out.structured_outputs.structural_tag is not None
    assert out.text is None
    grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
    assert _is_grammar_accept_string(
        grammar,
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>",
    )


def test_glm47_operator_level_overrides_parser_default(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools: list[ChatCompletionToolsParam],
):
    monkeypatch.setattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", True)

    class TestParser(DelegatingParser):
        tool_parser_cls = Glm47MoeModelToolParser
        tool_strict_level = ToolStrictLevel.PARAMETER

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=[tool.model_dump(exclude_none=True) for tool in sample_tools],
        tool_choice="required",
    )
    parser = TestParser(MagicMock(), tools=sample_tools)
    parser._reasoning_parser = MagicMock(adjust_request=lambda request: request)

    out = parser.adjust_request(request)

    assert out.structured_outputs is not None
    content = json.loads(out.structured_outputs.structural_tag)["format"]["tags"][0][
        "content"
    ]
    assert content["json_schema"] == sample_tools[0].function.parameters


def test_glm47_required_with_non_strict_tools_attaches_shallow_tag(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools: list[ChatCompletionToolsParam],
):
    monkeypatch.setattr(envs, "VLLM_ENFORCE_STRICT_TOOL_CALLING", True)

    class TestParser(DelegatingParser):
        tool_parser_cls = Glm47MoeModelToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=[tool.model_dump(exclude_none=True) for tool in sample_tools],
        tool_choice="required",
    )
    parser = TestParser(MagicMock(), tools=sample_tools)
    parser._reasoning_parser = MagicMock(adjust_request=lambda request: request)

    out = parser.adjust_request(request)

    assert out.structured_outputs is not None
    assert out.structured_outputs.json is None
    grammar = Grammar.from_structural_tag(out.structured_outputs.structural_tag)
    call = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>"
    )
    assert _is_grammar_accept_string(grammar, call)
    assert not _is_grammar_accept_string(grammar, "It is sunny in Paris.")
    assert not _is_grammar_accept_string(grammar, call.replace("city", "ctiy"))


@pytest.mark.parametrize("policy", ["auto", "required", "named"])
def test_mimo_strict_compact_xml(policy):
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "run",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer", "minimum": 1}},
                    "required": ["n"],
                    "additionalProperties": False,
                },
            },
        )
    ]
    choice = (
        ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name="run")
        )
        if policy == "named"
        else policy
    )
    tag = get_model_structural_tag("mimo", tools, choice, reasoning=False)
    grammar = Grammar.from_structural_tag(tag)
    call = "<tool_call><function=run><parameter=n>2</parameter></function></tool_call>"
    assert _is_grammar_accept_string(grammar, call)
    assert _is_grammar_accept_string(grammar, call * 2) == (policy != "named")
    assert _is_grammar_accept_string(grammar, "answer") == (policy == "auto")
    for invalid in [
        call.replace("function=run", "function=unknown"),
        call.replace("<parameter=n>2</parameter>", ""),
        call.replace(">2</parameter>", ">bad</parameter>"),
        call.replace(">2</parameter>", ">0</parameter>"),
        call.replace("<parameter=n>", "<parameter=other>"),
        call.replace("</function>", "<parameter=n>3</parameter></function>"),
        "<think>plan</think>" + call,
    ]:
        assert not _is_grammar_accept_string(grammar, invalid)
    tools[0].function.strict = False
    relaxed = Grammar.from_structural_tag(
        get_model_structural_tag("mimo", tools, "required", reasoning=False)
    )
    assert _is_grammar_accept_string(
        relaxed, call.replace("<parameter=n>", "<parameter=other>")
    )


_DS_CALLS_BEGIN = "<\uff5ctool\u2581calls\u2581begin\uff5c>"
_DS_CALL_BEGIN = "<\uff5ctool\u2581call\u2581begin\uff5c>"
_DS_SEP = "<\uff5ctool\u2581sep\uff5c>"
_DS_CALL_END = "<\uff5ctool\u2581call\u2581end\uff5c>"
_DS_CALLS_END = "<\uff5ctool\u2581calls\u2581end\uff5c>"

# One entry per tool-call list shape: top-level triggered tags (mimo, qwen3),
# tags with a separator (hermes required), and a calls block wrapping the list
# (deepseek). Each is (tool_parser, list prefix, one call, list suffix).
_SINGLE_CALL_CASES = [
    (
        "mimo",
        "",
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
        "</function></tool_call>",
        "",
    ),
    (
        "hermes",
        "",
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}'
        "\n</tool_call>",
        "",
    ),
    (
        "qwen3_xml",
        "",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
        "</parameter>\n</function>\n</tool_call>",
        "",
    ),
    (
        "deepseek_v31",
        _DS_CALLS_BEGIN,
        f'{_DS_CALL_BEGIN}get_weather{_DS_SEP}{{"city": "Paris"}}{_DS_CALL_END}',
        _DS_CALLS_END,
    ),
]

_SINGLE_CALL_IDS = [case[0] for case in _SINGLE_CALL_CASES]


def _tool_calling_grammar(
    tool_parser: str,
    tools: list[ChatCompletionToolsParam],
    request: ChatCompletionRequest | ResponsesRequest,
) -> Grammar | None:
    from vllm.parser.parser_manager import ParserManager

    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool_parser, enable_auto_tools=True
    )
    adjusted = parser_cls(MagicMock(), tools=tools).adjust_request(request)
    structured_outputs = adjusted.structured_outputs
    if structured_outputs is None or structured_outputs.structural_tag is None:
        return None
    return Grammar.from_structural_tag(structured_outputs.structural_tag)


@pytest.mark.parametrize("policy", ["auto", "required"])
@pytest.mark.parametrize(
    "tool_parser,prefix,call,suffix", _SINGLE_CALL_CASES, ids=_SINGLE_CALL_IDS
)
def test_parallel_tool_calls_false_limits_grammar_to_one_call(
    sample_tools, tool_parser, prefix, call, suffix, policy
):
    # Tools are not strict, so auto gets a grammar only because of the limit.
    request = ChatCompletionRequest(
        model="m",
        messages=[],
        tools=[t.model_dump() for t in sample_tools],
        tool_choice=policy,
        parallel_tool_calls=False,
    )

    grammar = _tool_calling_grammar(tool_parser, sample_tools, request)

    assert grammar is not None
    assert _is_grammar_accept_string(grammar, prefix + call + suffix)
    assert not _is_grammar_accept_string(grammar, prefix + call * 2 + suffix)
    assert not _is_grammar_accept_string(grammar, prefix + call + suffix + "text")


@pytest.mark.parametrize(
    "tool_parser,prefix,call,suffix", _SINGLE_CALL_CASES, ids=_SINGLE_CALL_IDS
)
def test_parallel_tool_calls_default_keeps_grammar_unlimited(
    sample_tools, tool_parser, prefix, call, suffix
):
    def request(policy: str) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="m",
            messages=[],
            tools=[t.model_dump() for t in sample_tools],
            tool_choice=policy,
        )

    assert _tool_calling_grammar(tool_parser, sample_tools, request("auto")) is None
    grammar = _tool_calling_grammar(tool_parser, sample_tools, request("required"))
    assert grammar is not None
    assert _is_grammar_accept_string(grammar, prefix + call * 2 + suffix)


def test_parallel_tool_calls_false_does_not_enable_calls_with_response_format(
    sample_tools,
):
    # With the flag unset, auto + response_format + non-strict tools is
    # format-only. Setting the flag to false must not make tool calls possible.
    request = ChatCompletionRequest(
        model="m",
        messages=[],
        tools=[t.model_dump() for t in sample_tools],
        tool_choice="auto",
        parallel_tool_calls=False,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        },
    )

    assert _tool_calling_grammar("hermes", sample_tools, request) is None


def test_responses_parallel_tool_calls_false_limits_grammar_to_one_call():
    # The Responses API has no post-hoc filter, so the grammar is the only limit.
    request = ResponsesRequest.model_validate(
        {
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
    )
    _, _, call, _ = _SINGLE_CALL_CASES[1]

    grammar = _tool_calling_grammar("hermes", request.tools, request)

    assert grammar is not None
    assert _is_grammar_accept_string(grammar, call)
    assert not _is_grammar_accept_string(grammar, call * 2)
