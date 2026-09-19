# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.parser.engine.registered_adapters import Gemma4ParserToolAdapter


class Gemma4EngineToolParser(Gemma4ParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Skip structured-output JSON for required tool choice.

        Gemma4 emits its native ``<|tool_call>call:...`` syntax, which the
        parser extracts directly. The base ``ToolParser.adjust_request`` would
        set ``structured_outputs`` for required/named and force JSON via guided
        decoding, conflicting with that native syntax (it leaks as content and
        crashes EngineCore under speculative decoding). Skip it so the model
        emits its native format (mirrors the GLM4 parser).

        Named tool choice cannot be enforced without that JSON constraint (and
        Gemma4 has no structural-tag equivalent), so reject it instead of
        silently falling back to auto parsing.
        """
        if request.tools:
            tc = request.tool_choice
            is_named = isinstance(
                tc, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            )
            if is_named:
                raise VLLMValidationError(
                    "Named tool choice is not supported for the Gemma 4 tool "
                    "parser because it cannot force a specific function call. "
                    'Use `tool_choice` set to "auto", "required", or "none".',
                    parameter="tool_choice",
                    value=tc,
                )
            if tc == "required":
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)
