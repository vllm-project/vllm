# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence

from openai.types.responses import FunctionTool, NamespaceTool, ToolChoiceFunction

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.glm_grammar import generate_glm_grammar
from vllm.tool_parsers.utils import Tool, iter_response_function_tool_info


class Glm47MoeModelToolParser(Glm47MoeParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if (
            not request.tools
            or request.tool_choice == "none"
            or envs.VLLM_ENFORCE_STRICT_TOOL_CALLING
            or request.structured_outputs is not None
            or self._has_strict_tools(request.tools)
            or self._has_response_format(request)
        ):
            return super().adjust_request(request)
        # Without structural tags (no strict tools, strict tool calling off),
        # constrain the full assistant turn with an EBNF grammar so
        # auto/required/named tool choices emit well-formed GLM XML.
        request.structured_outputs = StructuredOutputsParams(
            grammar=generate_glm_grammar(
                enable_thinking=self._enable_thinking(request),
                functions=self._grammar_functions(request),
            )
        )
        if isinstance(request, ResponsesRequest):
            request.text = None
        else:
            request.response_format = None
        request._grammar_from_parser = True
        request.skip_special_tokens = False
        return request

    @staticmethod
    def _has_strict_tools(tools: Sequence[Tool]) -> bool:
        for tool in tools:
            if isinstance(tool, FunctionTool) and tool.strict is True:
                return True
            if isinstance(tool, NamespaceTool) and any(
                getattr(t, "strict", None) is True for t in tool.tools
            ):
                return True
            if (
                isinstance(tool, ChatCompletionToolsParam)
                and tool.function.strict is True
            ):
                return True
        return False

    @staticmethod
    def _has_response_format(
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> bool:
        if isinstance(request, ResponsesRequest):
            return request.text is not None
        return (
            request.response_format is not None
            and request.response_format.type != "text"
        )

    @staticmethod
    def _function_specs(tools: Sequence[Tool]) -> list[tuple[str, dict]]:
        specs: list[tuple[str, dict]] = []
        for tool in tools:
            if isinstance(tool, (FunctionTool, NamespaceTool)):
                specs.extend(
                    (name, params or {})
                    for name, params in iter_response_function_tool_info(tool)
                )
            elif isinstance(tool, ChatCompletionToolsParam):
                specs.append((tool.function.name, tool.function.parameters or {}))
        return specs

    @classmethod
    def _grammar_functions(
        cls, request: ChatCompletionRequest | ResponsesRequest
    ) -> list[tuple[str, dict]]:
        specs = cls._function_specs(request.tools or [])
        tool_choice = request.tool_choice
        if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
            return [s for s in specs if s[0] == tool_choice.function.name]
        if isinstance(tool_choice, ToolChoiceFunction):
            return [s for s in specs if s[0] == tool_choice.name]
        return specs

    @staticmethod
    def _enable_thinking(request: ChatCompletionRequest | ResponsesRequest) -> bool:
        kwargs = request.chat_template_kwargs or {}
        for key in ("enable_thinking", "thinking"):
            if (value := kwargs.get(key)) is not None:
                return bool(value)
        effort = None
        if isinstance(request, ChatCompletionRequest):
            effort = request.reasoning_effort
        elif request.reasoning is not None:
            effort = request.reasoning.effort
        if effort is not None:
            return effort != "none"
        return True
