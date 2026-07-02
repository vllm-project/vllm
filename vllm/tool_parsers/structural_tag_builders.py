# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal, TypeAlias

from xgrammar import StructuralTag
from xgrammar.openai_tool_call_schema import (
    BuiltinToolParam,
    FunctionToolParam,
)
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

SimplifiedToolChoice: TypeAlias = Literal["auto", "required", "forced"]
THINK_EXCLUDES: list[str] = []  # "<tool_call>", "<tool_call>"


def _get_function_parameters(function) -> dict[str, Any] | bool:
    if getattr(function, "strict", None) is False:
        return True
    return function.parameters if function.parameters is not None else True


def _get_builtin_tool_name(tool: BuiltinToolParam) -> str:
    return tool.name if tool.name is not None else tool.type


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
                excludes=THINK_EXCLUDES,
            )
            if tags
            else AnyTextFormat(excludes=THINK_EXCLUDES)
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


def _json_schema_content(function, style=None):
    kwargs = {"json_schema": _get_function_parameters(function)}
    if style is not None:
        kwargs["style"] = style
    return JSONSchemaFormat(**kwargs)


def _simple_json_tags(tools, begin_fn, end, style=None):
    return [
        TagFormat(
            begin=begin_fn(tool.function.name),
            content=_json_schema_content(tool.function, style=style),
            end=end,
        )
        for tool in tools
    ]


def get_llama_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools, reasoning

    tags = _simple_json_tags(
        tools,
        lambda name: f'{{"name": "{name}", "parameters": ',
        "}",
    )

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=['{"name": '],
                tags=tags,
                excludes=THINK_EXCLUDES,
            )
            if tags
            else AnyTextFormat(excludes=THINK_EXCLUDES)
        )
    elif tool_choice == "forced":
        suffix_tag = tags[0]
    else:
        suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)

    return StructuralTag(format=suffix_tag)


def _kimi_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    return [
        TagFormat(
            begin=f"<|tool_call_begin|>functions.{tool.function.name}:",
            content=SequenceFormat(
                elements=[
                    RegexFormat(pattern=r"\d+"),
                    ConstStringFormat(value="<|tool_call_argument_begin|>"),
                    JSONSchemaFormat(
                        json_schema=_get_function_parameters(tool.function)
                    ),
                ]
            ),
            end="<|tool_call_end|>",
        )
        for tool in tools
    ]


def get_kimi_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools

    tags = _kimi_tool_tags(tools)

    if tool_choice == "auto":
        if tags:
            tool_calls = TagFormat(
                begin="<|tool_calls_section_begin|>",
                content=TagsWithSeparatorFormat(
                    tags=tags, separator="", at_least_one=True
                ),
                end="<|tool_calls_section_end|>",
            )
            suffix_tag = TriggeredTagsFormat(
                triggers=["<|tool_calls_section_begin|>"],
                tags=[tool_calls],
                excludes=[*THINK_EXCLUDES, "<|tool_call_begin|>"],
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDES)
    elif tool_choice == "forced":
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="<|tool_calls_section_begin|>"),
                tags[0],
                ConstStringFormat(value="<|tool_calls_section_end|>"),
            ]
        )
    else:
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="<|tool_calls_section_begin|>"),
                TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True),
                ConstStringFormat(value="<|tool_calls_section_end|>"),
            ]
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    return StructuralTag(
        format=SequenceFormat(
            elements=[
                TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                suffix_tag,
            ]
        )
    )


def get_deepseek_r1_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools

    tags = _simple_json_tags(
        tools,
        lambda name: f"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n",
        "\n```<｜tool▁call▁end｜>",
    )

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=["<｜tool▁calls▁begin｜>"],
                tags=[
                    TagFormat(
                        begin="<｜tool▁calls▁begin｜>",
                        content=TagsWithSeparatorFormat(
                            tags=tags, separator="\n", at_least_one=True
                        ),
                        end="<｜tool▁calls▁end｜>",
                    )
                ],
                excludes=THINK_EXCLUDES,
            )
            if tags
            else AnyTextFormat(excludes=THINK_EXCLUDES)
        )
    elif tool_choice == "forced":
        function = tools[0].function
        suffix_tag = TagFormat(
            begin=(
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
                f"<｜tool▁sep｜>{function.name}\n```json\n"
            ),
            content=JSONSchemaFormat(json_schema=_get_function_parameters(function)),
            end="\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        )
    else:
        suffix_tag = TagFormat(
            begin="<｜tool▁calls▁begin｜>",
            content=TagsWithSeparatorFormat(
                tags=tags, separator="\n", at_least_one=True
            ),
            end="<｜tool▁calls▁end｜>",
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    return StructuralTag(
        format=SequenceFormat(
            elements=[
                TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                suffix_tag,
            ]
        )
    )


def get_deepseek_v3_1_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools

    tags = _simple_json_tags(
        tools,
        lambda name: f"<｜tool▁call▁begin｜>{name}<｜tool▁sep｜>",
        "<｜tool▁call▁end｜>",
    )

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=["<｜tool▁calls▁begin｜>"],
                tags=[
                    TagFormat(
                        begin="<｜tool▁calls▁begin｜>",
                        content=TagsWithSeparatorFormat(
                            tags=tags, separator="", at_least_one=True
                        ),
                        end="<｜tool▁calls▁end｜>",
                    )
                ],
                excludes=THINK_EXCLUDES,
            )
            if tags
            else AnyTextFormat(excludes=THINK_EXCLUDES)
        )
    elif tool_choice == "forced":
        function = tools[0].function
        suffix_tag = TagFormat(
            begin=f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function.name}<｜tool▁sep｜>",
            content=JSONSchemaFormat(json_schema=_get_function_parameters(function)),
            end="<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        )
    else:
        suffix_tag = TagFormat(
            begin="<｜tool▁calls▁begin｜>",
            content=TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True),
            end="<｜tool▁calls▁end｜>",
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    return StructuralTag(
        format=SequenceFormat(
            elements=[
                TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                suffix_tag,
            ]
        )
    )


def _qwen_xml_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    return _simple_json_tags(
        tools,
        lambda name: f"<tool_call>\n<function={name}>\n",
        "\n</function>\n</tool_call>",
        style="qwen_xml",
    )


def get_qwen_3_5_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools

    tags = _qwen_xml_tool_tags(tools)

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=["<tool_call>\n<function="],
                tags=tags,
                excludes=THINK_EXCLUDES,
            )
            if tags
            else AnyTextFormat(excludes=THINK_EXCLUDES)
        )
    elif tool_choice == "forced":
        suffix_tag = tags[0]
    else:
        suffix_tag = TagsWithSeparatorFormat(
            tags=tags, separator="\n", at_least_one=True
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    return StructuralTag(
        format=SequenceFormat(
            elements=[
                SequenceFormat(
                    elements=[
                        TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                        ConstStringFormat(value="\n\n"),
                    ]
                ),
                suffix_tag,
            ]
        )
    )


get_qwen_3_coder_structural_tag = get_qwen_3_5_structural_tag


def get_qwen_3_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools

    tags = _simple_json_tags(
        tools,
        lambda name: f'<tool_call>\n{{"name": "{name}", "arguments": ',
        "}\n</tool_call>",
    )

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=["<tool_call>"],
                tags=tags,
                excludes=THINK_EXCLUDES,
            )
            if tags
            else AnyTextFormat(excludes=THINK_EXCLUDES)
        )
    elif tool_choice == "forced":
        suffix_tag = tags[0]
    else:
        suffix_tag = TagsWithSeparatorFormat(
            tags=tags, separator="\n", at_least_one=True
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    return StructuralTag(
        format=SequenceFormat(
            elements=[
                SequenceFormat(
                    elements=[
                        TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                        ConstStringFormat(value="\n\n"),
                    ]
                ),
                suffix_tag,
            ]
        )
    )


def _harmony_function_tool_tags(name, parameters):
    content = JSONSchemaFormat(json_schema=parameters)
    return [
        TagFormat(
            begin=(
                f"<|channel|>commentary to=functions.{name}<|constrain|>json<|message|>"
            ),
            content=content,
            end="<|call|>",
        ),
        TagFormat(
            begin=(
                f" to=functions.{name}<|channel|>commentary "
                "<|constrain|>json<|message|>"
            ),
            content=content,
            end="<|call|>",
        ),
        TagFormat(
            begin=f" to=functions.{name}<|channel|>commentary json<|message|>",
            content=content,
            end="<|call|>",
        ),
    ]


def _harmony_builtin_tool_tags(name, parameters):
    content = JSONSchemaFormat(json_schema=parameters)
    return [
        TagFormat(
            begin=f"<|channel|>commentary to={name} code<|message|>",
            content=content,
            end="<|call|>",
        ),
        TagFormat(
            begin=f" to={name}<|channel|>commentary code<|message|>",
            content=content,
            end="<|call|>",
        ),
    ]


def get_harmony_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    tags = []

    if tool_choice == "auto":
        for tool in tools:
            tags.extend(
                _harmony_function_tool_tags(
                    tool.function.name,
                    _get_function_parameters(tool.function),
                )
            )
        for tool in builtin_tools:
            tags.extend(
                _harmony_builtin_tool_tags(
                    _get_builtin_tool_name(tool),
                    _get_function_parameters(tool),
                )
            )
        tags.append(
            TagFormat(
                begin="<|channel|>final<|message|>",
                content=AnyTextFormat(),
                end=["<|end|>", "<|return|>"],
            )
        )
    elif tool_choice == "forced":
        if builtin_tools:
            tool = builtin_tools[0]
            tags.extend(
                _harmony_builtin_tool_tags(
                    _get_builtin_tool_name(tool),
                    _get_function_parameters(tool),
                )
            )
        else:
            tool = tools[0]
            tags.extend(
                _harmony_function_tool_tags(
                    tool.function.name,
                    _get_function_parameters(tool.function),
                )
            )
    else:
        for tool in builtin_tools:
            tags.extend(
                _harmony_builtin_tool_tags(
                    _get_builtin_tool_name(tool),
                    _get_function_parameters(tool),
                )
            )
        for tool in tools:
            tags.extend(
                _harmony_function_tool_tags(
                    tool.function.name,
                    _get_function_parameters(tool.function),
                )
            )

    if reasoning:
        tags.append(
            TagFormat(
                begin="<|channel|>analysis<|message|>",
                content=AnyTextFormat(),
                end=["<|end|>", "<|return|>"],
            )
        )

    return StructuralTag(
        format=TagsWithSeparatorFormat(tags=tags, separator="<|start|>assistant")
    )


def _deepseek_dsml_structural_tag(
    tools: list[FunctionToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    function_calls_begin: str,
    function_calls_end: str,
    function_calls_trigger: str,
) -> StructuralTag:
    tags = _simple_json_tags(
        tools,
        lambda name: f'<｜DSML｜invoke name="{name}">\n',
        "</｜DSML｜invoke>\n",
        style="deepseek_xml",
    )

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=[function_calls_trigger],
                tags=[
                    TagFormat(
                        begin=function_calls_begin,
                        content=TagsWithSeparatorFormat(
                            tags=tags, separator="", at_least_one=True
                        ),
                        end=function_calls_end,
                    )
                ],
                excludes=THINK_EXCLUDES,
            )
            if tags
            else AnyTextFormat(excludes=THINK_EXCLUDES)
        )
    elif tool_choice == "forced":
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n\n" + function_calls_begin),
                tags[0],
                ConstStringFormat(value=function_calls_end),
            ]
        )
    else:
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n\n" + function_calls_begin),
                TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True),
                ConstStringFormat(value=function_calls_end),
            ]
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    return StructuralTag(
        format=SequenceFormat(
            elements=[
                TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                suffix_tag,
            ]
        )
    )


def get_deepseek_v3_2_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools
    return _deepseek_dsml_structural_tag(
        tools,
        tool_choice,
        reasoning,
        function_calls_begin="<｜DSML｜function_calls>\n",
        function_calls_end="</｜DSML｜function_calls>",
        function_calls_trigger="<｜DSML｜function_calls>",
    )


def _glm_4_7_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    return _simple_json_tags(
        tools,
        lambda name: f"<tool_call>{name}",
        "</tool_call>",
        style="glm_xml",
    )


def get_glm_4_7_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools

    tags = _glm_4_7_tool_tags(tools)

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=["<tool_call>"],
                tags=tags,
                excludes=THINK_EXCLUDES,
            )
            if tags
            else AnyTextFormat(excludes=THINK_EXCLUDES)
        )
    elif tool_choice == "forced":
        suffix_tag = tags[0]
    else:
        suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    return StructuralTag(
        format=SequenceFormat(
            elements=[
                TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                suffix_tag,
            ]
        )
    )


def get_deepseek_v4_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
) -> StructuralTag:
    del builtin_tools
    return _deepseek_dsml_structural_tag(
        tools,
        tool_choice,
        reasoning,
        function_calls_begin="<｜DSML｜tool_calls>\n",
        function_calls_end="</｜DSML｜tool_calls>",
        function_calls_trigger="<｜DSML｜tool_calls>",
    )
