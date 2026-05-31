# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Model-specific structural tag builders adapted from XGrammar's
# builtin structural tag implementations:
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/builtin_structural_tag.py

from collections.abc import Callable
from typing import Any, Literal

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    RegexFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
)

SimplifiedToolChoice = Literal["auto", "required", "forced"]
ToolChoice = (
    Literal["none", "auto", "required"] | ChatCompletionNamedToolChoiceParam | None
)
StructuralTagBuilder = Callable[
    [list[ChatCompletionToolsParam], SimplifiedToolChoice, bool],
    StructuralTag,
]

_structural_tag_registry: dict[str, StructuralTagBuilder] = {}


def register_model_structural_tag(name: str):
    """Register a vLLM-owned model-specific structural tag builder."""

    def decorator(func: StructuralTagBuilder) -> StructuralTagBuilder:
        _structural_tag_registry[name] = func
        return func

    return decorator


def get_model_structural_tag(
    model: str,
    tools: list[ChatCompletionToolsParam] | None,
    tool_choice: ToolChoice,
    reasoning: bool,
) -> StructuralTag | None:
    """Build a structural tag from vLLM-owned model-specific builders."""

    builder = _structural_tag_registry.get(model)
    if builder is None:
        supported = list(_structural_tag_registry.keys())
        raise ValueError(f"Unknown format type: {model}, supported types: {supported}")

    normalized_tools, simplified_tool_choice = _normalize_tool_choice(
        tools=tools,
        tool_choice=tool_choice,
    )
    if not normalized_tools:
        return None

    return builder(normalized_tools, simplified_tool_choice, reasoning)


def _normalize_tool_choice(
    tools: list[ChatCompletionToolsParam] | None,
    tool_choice: ToolChoice,
) -> tuple[list[ChatCompletionToolsParam], SimplifiedToolChoice]:
    """Normalize vLLM ChatCompletion tool_choice for structural tag builders."""

    if not tools:
        return [], "auto"

    if tool_choice is None or tool_choice == "none":
        return [], "auto"

    if tool_choice == "auto":
        return tools, "auto"

    if tool_choice == "required":
        return tools, "required"

    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        tool_name = tool_choice.function.name
        filtered_tools = [tool for tool in tools if tool.function.name == tool_name]
        if not filtered_tools:
            raise ValueError(
                f"The tool with name '{tool_name}' is not found in the tools list."
            )
        return filtered_tools, "forced"

    raise ValueError(f"Unsupported tool_choice for structural tag: {tool_choice}")


def _get_function_parameters(function: Any) -> dict[str, Any] | bool:
    """Return the JSON schema used for constrained tool arguments."""

    if getattr(function, "strict", None) is False:
        return True
    if function.parameters is None:
        return True
    return function.parameters


def _get_xgrammar_supported_function_parameters(function: Any) -> dict[str, Any] | bool:
    """Return function parameters after applying xgrammar compatibility checks."""

    parameters = _get_function_parameters(function)
    if isinstance(parameters, dict) and has_xgrammar_unsupported_json_features(
        parameters
    ):
        raise ValueError(
            "The provided JSON schema contains features not supported by xgrammar."
        )
    return parameters


_enable_structured_outputs_in_reasoning: bool = False


def set_enable_structured_outputs_in_reasoning(enabled: bool) -> None:
    """Publish the engine's ``enable_in_reasoning`` flag to tool parsers.

    Called once during APIServer startup so request-time parsers can read
    it without going through the EngineCore-only contextvar.
    """

    global _enable_structured_outputs_in_reasoning
    _enable_structured_outputs_in_reasoning = bool(enabled)


def get_enable_structured_outputs_in_reasoning() -> bool:
    """Whether structured outputs are active during the reasoning phase.

    When ``True``, the structural tag will cover the reasoning part:
    ``<think>...</think>`` prefix (if available); when ``False`` (default), the tag only
    constrains the post-reasoning suffix.
    """

    return _enable_structured_outputs_in_reasoning


@register_model_structural_tag("deepseek_v4")
def get_deepseek_v4_structural_tag(
    tools: list[ChatCompletionToolsParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    """Build DeepSeek V4 structural tags."""

    invoke_begin_prefix = '<｜DSML｜invoke name="'
    invoke_begin_suffix = '">\n'
    invoke_end = "</｜DSML｜invoke>\n"
    tool_calls_prefix = "\n\n"
    function_calls_begin = "<｜DSML｜tool_calls>\n"
    function_calls_end = "</｜DSML｜tool_calls>"
    function_calls_trigger = "<｜DSML｜tool_calls>"
    think_tag_end = "</think>"
    think_exclude_tokens = ["<think>", "</think>"]
    xml_style = "deepseek_xml"

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            tags.append(
                TagFormat(
                    begin=invoke_begin_prefix + function.name + invoke_begin_suffix,
                    content=JSONSchemaFormat(
                        json_schema=parameters,
                        style=xml_style,
                    ),
                    end=invoke_end,
                )
            )

        if tags:
            function_calling_tags = TagsWithSeparatorFormat(
                tags=tags,
                separator="\n",
                at_least_one=True,
            )
            suffix_tag = TriggeredTagsFormat(
                triggers=[function_calls_trigger],
                tags=[
                    TagFormat(
                        begin=function_calls_begin,
                        content=function_calling_tags,
                        end=function_calls_end,
                    )
                ],
                excludes=think_exclude_tokens,
            )
        else:
            suffix_tag = AnyTextFormat(excludes=think_exclude_tokens)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=tool_calls_prefix + function_calls_begin),
                TagFormat(
                    begin=invoke_begin_prefix + function.name + invoke_begin_suffix,
                    content=JSONSchemaFormat(
                        json_schema=_get_function_parameters(function),
                        style=xml_style,
                    ),
                    end=invoke_end,
                ),
                ConstStringFormat(value=function_calls_end),
            ]
        )

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            tags.append(
                TagFormat(
                    begin=invoke_begin_prefix + function.name + invoke_begin_suffix,
                    content=JSONSchemaFormat(
                        json_schema=parameters,
                        style=xml_style,
                    ),
                    end=invoke_end,
                )
            )
        assert len(tags) > 0
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=tool_calls_prefix + function_calls_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="\n",
                    at_least_one=True,
                ),
                ConstStringFormat(value=function_calls_end),
            ]
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=think_tag_end)
    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_model_structural_tag("kimi_k2")
def get_kimi_k2_structural_tag(
    tools: list[ChatCompletionToolsParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    """Build Kimi K2 native tool-call structural tags."""

    section_begin = "<|tool_calls_section_begin|>"
    section_end = "<|tool_calls_section_end|>"
    tool_call_begin = "<|tool_call_begin|>"
    argument_begin = "<|tool_call_argument_begin|>"
    tool_call_end = "<|tool_call_end|>"
    think_tag_end = "</think>"
    think_exclude_tokens = ["<think>", "</think>"]
    whitespace = RegexFormat(pattern=r"\s*")

    tags = []
    for tool in tools:
        function = tool.function
        tags.append(
            TagFormat(
                begin=tool_call_begin,
                content=SequenceFormat(
                    elements=[
                        whitespace,
                        ConstStringFormat(value=f"functions.{function.name}:"),
                        RegexFormat(pattern=r"\d+"),
                        whitespace,
                        ConstStringFormat(value=argument_begin),
                        whitespace,
                        JSONSchemaFormat(
                            json_schema=_get_xgrammar_supported_function_parameters(
                                function
                            ),
                        ),
                        whitespace,
                    ]
                ),
                end=tool_call_end,
            )
        )

    if tool_choice == "auto":
        suffix_tag = TriggeredTagsFormat(
            triggers=[section_begin],
            tags=[
                TagFormat(
                    begin=section_begin,
                    content=SequenceFormat(
                        elements=[
                            whitespace,
                            TagsWithSeparatorFormat(
                                tags=tags,
                                separator="",
                                at_least_one=True,
                            ),
                            whitespace,
                        ]
                    ),
                    end=section_end,
                )
            ],
            excludes=[*think_exclude_tokens, tool_call_begin],
        )

    elif tool_choice == "forced":
        if len(tags) != 1:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=section_begin),
                whitespace,
                tags[0],
                whitespace,
                ConstStringFormat(value=section_end),
            ]
        )

    elif tool_choice == "required":
        if not tags:
            raise ValueError("Required tool choice must resolve to at least one tool.")
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=section_begin),
                whitespace,
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                ),
                whitespace,
                ConstStringFormat(value=section_end),
            ]
        )
    else:
        raise ValueError(f"Unsupported Kimi K2 tool choice: {tool_choice}")

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=think_tag_end)
    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_model_structural_tag("qwen_3_5")
def get_qwen_3_5_structural_tag(
    tools: list[ChatCompletionToolsParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    """Build Qwen XML structural tags.

    This format is used for Qwen3-Coder/Qwen3.5/Qwen3.6 and is compatible with
    Qwen variants that use the same XML tool-call format.
    """
    tool_call_begin_prefix = "<tool_call>\n<function="
    tool_call_begin_suffix = ">\n"
    tool_call_end = "\n</function>\n</tool_call>"
    tool_call_trigger = "<tool_call>\n<function="
    think_tag_end = "</think>"
    think_suffix = "\n\n"
    think_exclude_tokens = ["<think>", "</think>"]

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            tags.append(
                TagFormat(
                    begin=f"{tool_call_begin_prefix}{function.name}{tool_call_begin_suffix}",
                    content=JSONSchemaFormat(json_schema=parameters, style="qwen_xml"),
                    end=tool_call_end,
                )
            )

        if tags:
            suffix_tag = TriggeredTagsFormat(
                triggers=[tool_call_trigger],
                tags=tags,
                excludes=think_exclude_tokens,
            )
        else:
            suffix_tag = AnyTextFormat(excludes=think_exclude_tokens)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = TagFormat(
            begin=f"{tool_call_begin_prefix}{function.name}{tool_call_begin_suffix}",
            content=JSONSchemaFormat(
                json_schema=_get_function_parameters(function),
                style="qwen_xml",
            ),
            end=tool_call_end,
        )

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            tags.append(
                TagFormat(
                    begin=f"{tool_call_begin_prefix}{function.name}{tool_call_begin_suffix}",
                    content=JSONSchemaFormat(json_schema=parameters, style="qwen_xml"),
                    end=tool_call_end,
                )
            )
        assert len(tags) > 0
        suffix_tag = TagsWithSeparatorFormat(
            tags=tags,
            separator="",
            at_least_one=True,
        )

    if not reasoning:
        result = StructuralTag(format=suffix_tag)
    else:
        prefix_tag = SequenceFormat(
            elements=[
                TagFormat(begin="", content=AnyTextFormat(), end=think_tag_end),
                ConstStringFormat(value=think_suffix),
            ]
        )
        result = StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))

    return result
