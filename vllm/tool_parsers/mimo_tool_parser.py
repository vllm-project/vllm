# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from xgrammar import get_model_structural_tag as get_xgrammar_structural_tag
from xgrammar.openai_tool_call_schema import BuiltinToolParam, FunctionToolParam
from xgrammar.structural_tag import (
    AnyTextFormat,
    JSONSchemaFormat,
    StructuralTag,
    TagFormat,
    TriggeredTagsFormat,
)

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.mimo import MiMoParser
from vllm.tool_parsers.structural_tag_registry import (
    SimplifiedToolChoice,
    register_vllm_structural_tag,
)
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

_, MiMoParserToolAdapter = make_adapters(MiMoParser)


class MiMoToolParser(MiMoParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "mimo"

    def get_structural_tag(
        self, request, *, reasoning=False, strict_level=ToolStrictLevel.AUTO
    ):
        tag = super().get_structural_tag(
            request, reasoning=reasoning, strict_level=strict_level
        )
        if (
            tag is not None
            and request.parallel_tool_calls is False
            and isinstance(tag.format, TriggeredTagsFormat)
        ):
            tag.format.stop_after_first = True
        return tag


@register_vllm_structural_tag("mimo")
def get_mimo_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    if token_suffix:
        raise ValueError("MiMo uses fixed tokens and cannot apply token_suffix")
    choice = (
        {"type": "function", "function": {"name": tools[0].function.name}}
        if tool_choice == "forced"
        else tool_choice
    )
    try:
        # Serving owns the reasoning boundary; constrain the tool/text suffix only.
        return get_xgrammar_structural_tag(
            "mimo",
            tools=[*tools, *builtin_tools],
            tool_choice=choice,
            reasoning="disabled",
        )
    except ValueError as error:
        if not str(error).startswith("Unknown format type: mimo,"):
            raise

    # XGrammar 0.2.7 has qwen_xml, but predates the compact MiMo builtin.
    if builtin_tools:
        raise ValueError("MiMo does not support builtin tools.")
    tags = [
        TagFormat(
            begin=f"<tool_call><function={tool.function.name}>",
            content=JSONSchemaFormat(
                json_schema=(
                    tool.function.parameters
                    if tool.function.strict is not False and tool.function.parameters
                    else True
                ),
                style="qwen_xml",
            ),
            end="</function></tool_call>",
        )
        for tool in tools
    ]
    excludes = ["<think>", "</think>", "</tool_call>", "<function="]
    if tool_choice == "forced":
        suffix = tags[0]
    elif tags:
        suffix = TriggeredTagsFormat(
            triggers=["<tool_call>"],
            tags=tags,
            excludes=excludes,
            at_least_one=tool_choice == "required",
        )
    else:
        suffix = AnyTextFormat(excludes=["<tool_call>", *excludes])
    return StructuralTag(format=suffix)
