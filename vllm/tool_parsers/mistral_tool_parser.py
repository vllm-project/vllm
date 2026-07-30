# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.parser.engine.registered_adapters import MistralParserToolAdapter

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MistralToolParser(MistralParserToolAdapter):  # type: ignore[valid-type, misc]
    # Marked so is_mistral_tool_parser() in vllm.utils.mistral recognises
    # this adapter and lets adjust_request() fire for tool_choice="none"
    # on grammar-capable Mistral tokenizers.
    IS_MISTRAL_TOOL_PARSER = True

    # Mistral emits tool calls in the [TOOL_CALLS]name[ARGS]{...} format, not the
    # plain JSON array / bare-args that the serving layer's generic required and
    # named handlers expect. Route required/named through this parser's own
    # extraction (treated like "auto") so the [TOOL_CALLS] format is parsed
    # correctly instead of crashing or dumping the raw envelope into arguments.
    supports_required_and_named = False

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        # Skip the base ToolParser tool_choice -> structured_outputs
        # conversion. Mistral enforces tool_choice via the grammar ``mode``
        # (auto/none/required/named); forwarding a tool-derived json_schema
        # would be unioned into the grammar by mistral-common, allowing raw
        # JSON in place of a real [TOOL_CALLS] call. Genuine user-provided
        # structured output on the request is still honored by the engine's
        # adjust_request.
        return self._parser_engine.adjust_request(request)
