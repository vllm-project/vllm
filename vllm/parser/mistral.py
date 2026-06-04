# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import FunctionCall
from vllm.parser.abstract_parser import DelegatingParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MistralParser(DelegatingParser):
    def __init__(self, tokenizer, tools=None, *args, **kwargs):
        super().__init__(tokenizer, tools, *args, **kwargs)
        from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

        if not isinstance(self._tool_parser, MistralToolParser):
            raise ValueError(
                "MistralParser requires --tool-call-parser mistral, "
                f"got {self._tool_parser.__class__.__name__}."
            )

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        content: str | None = model_output
        reasoning, tool_calls = None, None

        # No grammar (pre-v11): delegate to base class for tool choice
        # parsing (named/required/auto).
        if not getattr(request, "_grammar_from_tool_parser", False):
            reasoning, content, tool_calls = super().parse(
                model_output, request, enable_auto_tools
            )
            if tool_calls:
                from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

                # Named/required tool_choice builds FunctionCalls without
                # ID, backfill with Mistral-format IDs.
                for tc in tool_calls:
                    if not tc.id:
                        tc.id = MistralToolCall.generate_random_id()

        # Grammar (v11+): output is already structured, use
        # MistralToolParser.extract_tool_calls directly.
        else:
            if self._reasoning_parser is not None:
                reasoning, content = self._reasoning_parser.extract_reasoning(
                    model_output, request
                )

            assert self._tool_parser is not None
            tool_call_info = self._tool_parser.extract_tool_calls(
                content if content is not None else "",
                request=request,  # type: ignore[arg-type]
            )
            if tool_call_info.tools_called:
                tool_calls = [
                    FunctionCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in tool_call_info.tool_calls
                ]
                content = tool_call_info.content

        return reasoning, content, tool_calls
