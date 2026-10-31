# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai.types.responses import ToolChoiceCustom, ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import Qwen3ParserToolAdapter


class Qwen3EngineToolParser(Qwen3ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "qwen_3_coder"
    supports_required_and_named = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Skip structured-output JSON for required/named tool choice.

        Qwen3 emits its native XML ``<tool_call><function=...`` syntax, which
        the parser extracts directly. The base ``ToolParser.adjust_request``
        would set ``structured_outputs`` for required/named and force JSON via
        guided decoding, conflicting with that native syntax (it leaks as
        content and crashes EngineCore under speculative decoding). Skip it so
        the model emits its native format (mirrors the Gemma4 and GLM4 parsers).
        """
        structured_outputs = getattr(request, "structured_outputs", None)
        if (
            structured_outputs is not None
            and structured_outputs.structural_tag is not None
        ):
            return request

        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(
                tc,
                (
                    ChatCompletionNamedToolChoiceParam,
                    ToolChoiceFunction,
                    ToolChoiceCustom,
                ),
            ):
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)
