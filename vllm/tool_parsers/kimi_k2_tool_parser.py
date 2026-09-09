# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import KimiK2ParserToolAdapter


class KimiK2ToolParser(KimiK2ParserToolAdapter):  # type: ignore[valid-type, misc]
    """Legacy tool-parser name backed by the Kimi K2 parser engine."""

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        skip_special_tokens = request.skip_special_tokens
        request = super().adjust_request(request)
        if not request.tools or request.tool_choice == "none":
            request.skip_special_tokens = skip_special_tokens
        return request
