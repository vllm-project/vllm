# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from xgrammar import StructuralTag
from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)

ToolChoice = (
    Literal["none", "auto", "required"] | ChatCompletionNamedToolChoiceParam | None
)

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
        "minimax",
        "glm_4_7",
        "deepseek_v4",
    }
)


def get_model_structural_tag(
    model: str,
    tools: list[ChatCompletionToolsParam] | None,
    tool_choice: ToolChoice,
    reasoning: bool,
) -> StructuralTag | None:
    """Build a structural tag with xgrammar's builtin model templates."""

    if not tools or tool_choice == "none":
        return None

    if model not in XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS:
        supported = sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS)
        raise ValueError(f"Unknown format type: {model}, supported types: {supported}")

    return get_xgrammar_model_structural_tag(
        model=model,
        tools=[_model_dump(tool) for tool in tools],
        tool_choice=_model_dump(tool_choice),
        reasoning=reasoning,
    )


def _model_dump(value: Any) -> Any:
    """Convert vLLM/Pydantic request objects to xgrammar's dict protocol."""

    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value
