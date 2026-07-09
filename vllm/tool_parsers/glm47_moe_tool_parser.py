# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter


class Glm47MoeModelToolParser(Glm47MoeParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(
                tc, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            ):
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)
