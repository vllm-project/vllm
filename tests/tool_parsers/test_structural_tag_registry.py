# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from xgrammar import StructuralTag

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
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser
from vllm.tool_parsers.llama_tool_parser import Llama3JsonToolParser
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.structural_tag_registry import (
    SUPPORTED_STRUCTURAL_TAG_MODELS,
    VLLM_BUILTIN_STRUCTURAL_TAG_MODELS,
    XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS,
    _get_function_parameters,
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
        (DeepSeekV32ToolParser, "deepseek_v3_2"),
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

    assert _get_function_parameters(function) is True
