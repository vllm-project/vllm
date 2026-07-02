# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
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

SUPPORTED_STRUCTURAL_TAG_MODELS = frozenset(
    {
        "deepseek_r1",
        "deepseek_v3_1",
        "deepseek_v3_2",
        "deepseek_v4",
        "glm_4_7",
        "harmony",
        "hermes",
        "kimi",
        "llama",
        "minimax",
        "qwen_3",
        "qwen_3_5",
        "qwen_3_coder",
    }
)


_STRUCTURAL_TAG_BUILDERS_MODULE = "vllm.tool_parsers.structural_tag_builders"

_VLLM_STRUCTURAL_TAG_REGISTRY: dict[str, StructuralTagBuilder] = {}
_VLLM_STRUCTURAL_TAG_LAZY_REGISTRY: dict[str, tuple[str, str]] = {
    "deepseek_r1": (
        _STRUCTURAL_TAG_BUILDERS_MODULE,
        "get_deepseek_r1_structural_tag",
    ),
    "deepseek_v3_1": (
        _STRUCTURAL_TAG_BUILDERS_MODULE,
        "get_deepseek_v3_1_structural_tag",
    ),
    "deepseek_v3_2": (
        _STRUCTURAL_TAG_BUILDERS_MODULE,
        "get_deepseek_v3_2_structural_tag",
    ),
    "deepseek_v4": (
        _STRUCTURAL_TAG_BUILDERS_MODULE,
        "get_deepseek_v4_structural_tag",
    ),
    "glm_4_7": (_STRUCTURAL_TAG_BUILDERS_MODULE, "get_glm_4_7_structural_tag"),
    "harmony": (_STRUCTURAL_TAG_BUILDERS_MODULE, "get_harmony_structural_tag"),
    "hermes": (_STRUCTURAL_TAG_BUILDERS_MODULE, "get_hermes_structural_tag"),
    "kimi": (_STRUCTURAL_TAG_BUILDERS_MODULE, "get_kimi_structural_tag"),
    "llama": (_STRUCTURAL_TAG_BUILDERS_MODULE, "get_llama_structural_tag"),
    "minimax": (_STRUCTURAL_TAG_BUILDERS_MODULE, "get_minimax_structural_tag"),
    "qwen_3": (_STRUCTURAL_TAG_BUILDERS_MODULE, "get_qwen_3_structural_tag"),
    "qwen_3_coder": (
        _STRUCTURAL_TAG_BUILDERS_MODULE,
        "get_qwen_3_5_structural_tag",
    ),
}


def _load_vllm_structural_tag_builder(model: str) -> StructuralTagBuilder | None:
    """Load and cache a lazily registered vLLM structural tag builder."""
    if model in _VLLM_STRUCTURAL_TAG_REGISTRY:
        return _VLLM_STRUCTURAL_TAG_REGISTRY[model]

    if model not in _VLLM_STRUCTURAL_TAG_LAZY_REGISTRY:
        return None

    module_path, function_name = _VLLM_STRUCTURAL_TAG_LAZY_REGISTRY[model]
    module = importlib.import_module(module_path)
    builder = getattr(module, function_name)
    if not callable(builder):
        raise TypeError(
            f"{function_name} in {module_path} is not a structural tag builder."
        )
    _VLLM_STRUCTURAL_TAG_REGISTRY[model] = builder
    return builder


def _any_tool_strict(
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool],
) -> bool:
    for tool in tools:
        if isinstance(tool, FunctionTool) and tool.strict is True:
            return True
        if isinstance(tool, ChatCompletionToolsParam) and tool.function.strict is True:
            return True
    return False


def get_model_structural_tag(
    model: str,
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool] | None,
    tool_choice: ToolChoice,
    reasoning: bool,
) -> StructuralTag | None:
    """Build a structural tag with xgrammar's builtin model templates."""

    if not tools or tool_choice == "none":
        return None

    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None

    dumped_tools = [_dump_tool_for_xgrammar(tool) for tool in tools]
    dumped_tool_choice = _dump_tool_choice_for_xgrammar(tool_choice)
    builder = _load_vllm_structural_tag_builder(model)
    if builder is not None:
        function_tools, builtin_tools, simplified_tool_choice = normalize_tool_choice(
            dumped_tools,
            dumped_tool_choice,
        )
        return builder(
            function_tools,
            builtin_tools,
            simplified_tool_choice,
            reasoning,
        )
    if model not in SUPPORTED_STRUCTURAL_TAG_MODELS:
        supported = sorted(
            set(SUPPORTED_STRUCTURAL_TAG_MODELS)
            | set(_VLLM_STRUCTURAL_TAG_REGISTRY)
            | set(_VLLM_STRUCTURAL_TAG_LAZY_REGISTRY)
        )
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
