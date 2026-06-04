# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Model-specific structural tag builders adapted from XGrammar's
# builtin structural tag implementations:
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/builtin_structural_tag.py

from typing import Literal

from xgrammar import StructuralTag
from xgrammar import get_model_structural_tag as _get_model_structural_tag
from xgrammar.openai_tool_call_schema import FunctionToolParam, NamedToolChoiceParam

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)

ToolChoice = (
    Literal["none", "auto", "required"] | ChatCompletionNamedToolChoiceParam | None
)


def get_model_structural_tag(
    model: str,
    tools: list[ChatCompletionToolsParam] | None,
    tool_choice: ToolChoice,
    reasoning: bool,
) -> StructuralTag | None:
    """Import a structural tag from xgrammar's built-in structural tag templates."""
    # Transform vllm's tool schema to xgrammar's tool schema.
    if tools:
        tools = [FunctionToolParam.model_validate(tool.model_dump()) for tool in tools]
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        tool_choice = NamedToolChoiceParam.model_validate(tool_choice.model_dump())

    return _get_model_structural_tag(
        model,
        tools=tools,
        tool_choice=tool_choice,
        reasoning=reasoning,
    )


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
