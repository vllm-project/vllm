# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

from openai.types.responses import FunctionTool
from openai.types.responses.response import ToolChoice as ResponsesToolChoice
from openai.types.responses.tool import Tool as ResponsesTool
from openai.types.responses.tool_choice_allowed import ToolChoiceAllowed
from openai.types.responses.tool_choice_function import ToolChoiceFunction
from xgrammar import StructuralTag, normalize_tool_choice
from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
from xgrammar.openai_tool_call_schema import (
    BuiltinToolParam,
    FunctionToolParam,
)
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)

ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"]
    | ChatCompletionNamedToolChoiceParam
    | ResponsesToolChoice
    | None
)
AllowedToolRef: TypeAlias = dict[str, object]
SimplifiedToolChoice: TypeAlias = Literal["auto", "required", "forced"]
StructuralTagBuilder: TypeAlias = Callable[
    [
        list[FunctionToolParam],
        list[BuiltinToolParam],
        SimplifiedToolChoice,
        bool,
    ],
    StructuralTag,
]

# Keep this list in sync with xgrammar.builtin_structural_tag. It is used for
# vLLM-side validation and for documenting the xgrammar builtin surface that
# can be requested by tool parsers through ``structural_tag_model``.
XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset(
    {
        "llama",
        "kimi",
        "deepseek_r1",
        "deepseek_v3_1",
        "qwen_3_5",
        "qwen_3_coder",
        "qwen_3",
        "harmony",
        "deepseek_v3_2",
        "glm_4_7",
        "deepseek_v4",
    }
)
VLLM_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset({"hermes"})
SUPPORTED_STRUCTURAL_TAG_MODELS = (
    XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS | VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
)

_VLLM_STRUCTURAL_TAG_REGISTRY: dict[str, StructuralTagBuilder] = {}


def register_vllm_structural_tag(model: str):
    """Register a vLLM-owned structural tag builder."""

    def decorator(func: StructuralTagBuilder) -> StructuralTagBuilder:
        _VLLM_STRUCTURAL_TAG_REGISTRY[model] = func
        return func

    return decorator


def get_model_structural_tag(
    model: str,
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool] | None,
    tool_choice: ToolChoice,
    reasoning: bool,
) -> StructuralTag | None:
    """Build a structural tag with xgrammar's builtin model templates."""

    if not tools or tool_choice == "none":
        return None

    dumped_tools = [_dump_tool_for_xgrammar(tool) for tool in tools]
    dumped_tool_choice = _dump_tool_choice_for_xgrammar(tool_choice)

    if model in _VLLM_STRUCTURAL_TAG_REGISTRY:
        function_tools, builtin_tools, simplified_tool_choice = normalize_tool_choice(
            dumped_tools,
            dumped_tool_choice,
        )
        return _VLLM_STRUCTURAL_TAG_REGISTRY[model](
            function_tools,
            builtin_tools,
            simplified_tool_choice,
            reasoning,
        )

    if model not in XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS:
        supported = sorted(SUPPORTED_STRUCTURAL_TAG_MODELS)
        raise ValueError(f"Unknown format type: {model}, supported types: {supported}")

    return get_xgrammar_model_structural_tag(
        model=model,
        tools=dumped_tools,
        tool_choice=dumped_tool_choice,
        reasoning=reasoning,
    )


def _dump_tool_for_xgrammar(
    tool: ChatCompletionToolsParam | ResponsesTool,
) -> dict[str, Any]:
    """Convert tool objects to xgrammar's Chat Completions tool protocol."""

    if isinstance(tool, FunctionTool):
        function: dict[str, Any] = {"name": tool.name}
        if tool.description is not None:
            function["description"] = tool.description
        if tool.parameters is not None:
            function["parameters"] = tool.parameters
        if tool.strict is not None:
            function["strict"] = tool.strict
        return {"type": "function", "function": function}
    dumped_tool = tool.model_dump(mode="json", exclude_none=True)
    if isinstance(tool, ChatCompletionToolsParam):
        return dumped_tool
    return dict(dumped_tool)


def _dump_tool_choice_for_xgrammar(
    tool_choice: ToolChoice,
) -> dict[str, Any] | str | None:
    """Convert tool_choice objects to xgrammar's expected protocol."""

    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        return tool_choice

    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.model_dump(mode="json", exclude_none=True)

    if isinstance(tool_choice, ToolChoiceFunction):
        return {
            "type": "function",
            "function": {"name": tool_choice.name},
        }

    if isinstance(tool_choice, ToolChoiceAllowed):
        return {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": tool_choice.mode,
                "tools": [
                    _dump_allowed_tool_ref_for_xgrammar(tool)
                    for tool in tool_choice.tools
                ],
            },
        }

    return tool_choice.model_dump(mode="json", exclude_none=True)


def _dump_allowed_tool_ref_for_xgrammar(tool_ref: AllowedToolRef) -> AllowedToolRef:
    if (
        tool_ref.get("type") == "function"
        and "function" not in tool_ref
        and "name" in tool_ref
    ):
        return {
            "type": "function",
            "function": {"name": tool_ref["name"]},
        }
    return tool_ref


def _get_function_parameters(function) -> dict[str, Any] | bool:
    if getattr(function, "strict", None) is False:
        return True
    return function.parameters if function.parameters is not None else True


def _hermes_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    arguments_field_prefix = '", "arguments": '
    formats = [
        # <tool_call>
        # {"name": "t1", "arguments": {"q": "v"}}
        # </tool_call>
        ('<tool_call>\n{"name": "', "}\n</tool_call>"),
        # <tool_call>{"name": "t1", "arguments": {"q": "v"}}</tool_call>
        ('<tool_call>{"name": "', "}</tool_call>"),
    ]

    return [
        TagFormat(
            begin=begin + tool.function.name + arguments_field_prefix,
            content=JSONSchemaFormat(
                json_schema=_get_function_parameters(tool.function)
            ),
            end=end,
        )
        for tool in tools
        for begin, end in formats
    ]


@register_vllm_structural_tag("hermes")
def get_hermes_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools, reasoning

    tool_call_trigger = "<tool_call>"

    if tool_choice == "auto":
        tags = _hermes_tool_tags(tools)
        suffix_tag = (
            TriggeredTagsFormat(triggers=[tool_call_trigger], tags=tags)
            if tags
            else AnyTextFormat()
        )
    elif tool_choice == "forced":
        suffix_tag = TagsWithSeparatorFormat(
            tags=_hermes_tool_tags(tools),
            separator="",
            at_least_one=True,
            stop_after_first=True,
        )
    else:
        suffix_tag = TagsWithSeparatorFormat(
            tags=_hermes_tool_tags(tools),
            separator="",
            at_least_one=True,
        )

    return StructuralTag(format=suffix_tag)


def _minimax_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    return [
        TagFormat(
            begin=f'<invoke name="{tool.function.name}">\n',
            content=JSONSchemaFormat(
                json_schema=_get_function_parameters(tool.function),
                style="minimax_xml",
            ),
            end="</invoke>\n",
        )
        for tool in tools
    ]


@register_vllm_structural_tag("minimax")
def get_minimax_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools, reasoning

    tool_call_begin = "<minimax:tool_call>\n"
    tool_call_end = "</minimax:tool_call>"
    tool_call_trigger = "<minimax:tool_call>"

    tags = _minimax_tool_tags(tools)

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=[tool_call_trigger],
                tags=[
                    TagFormat(
                        begin=tool_call_begin,
                        content=TagsWithSeparatorFormat(
                            tags=tags,
                            separator="",
                            at_least_one=True,
                        ),
                        end=tool_call_end,
                    )
                ],
                excludes=["<think>", "</think>"],
            )
            if tags
            else AnyTextFormat(excludes=["<think>", "</think>"])
        )
    elif tool_choice == "forced":
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n" + tool_call_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                    stop_after_first=True,
                ),
                ConstStringFormat(value=tool_call_end),
            ]
        )
    else:
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n" + tool_call_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                ),
                ConstStringFormat(value=tool_call_end),
            ]
        )

    return StructuralTag(format=suffix_tag)
