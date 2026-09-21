# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections.abc import Sequence

from openai.types.responses import FunctionTool, NamespaceTool

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.utils import Tool


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
        # attach the non-strict structural tag so auto/required/named tool
        # choices still emit well-formed GLM XML. The dedicated registration
        # key keeps the strict glm_4_7 path on the xgrammar builtin.
        from vllm.tool_parsers.structural_tag_registry import (
            get_model_structural_tag,
        )
        from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

        # FUNCTION lifts the auto/no-strict gate so envelope constraints still
        # apply without pinning argument schemas (PARAMETER).
        structural_tag = get_model_structural_tag(
            model="glm_4_7_nonstrict",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=False,
            strict_level=ToolStrictLevel.FUNCTION,
        )
        if structural_tag is None:
            return super().adjust_request(request)
        request.structured_outputs = StructuredOutputsParams(
            structural_tag=json.dumps(structural_tag.model_dump())
        )
        if isinstance(request, ResponsesRequest):
            request.text = None
        else:
            request.response_format = None
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
