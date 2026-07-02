# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage, FunctionCall
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

    def _maybe_force_auto_tool_parsing(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> None:
        # When the Mistral grammar factory injected structured outputs,
        # the model emits v11+ format ([TOOL_CALLS]name{args}) that the
        # named/required parsers can't handle. Disable them so all
        # tool_choice modes fall back to auto tool parsing via
        # extract_tool_calls.
        if getattr(request, "_grammar_from_tool_parser", False):
            assert self._tool_parser is not None
            self._tool_parser.supports_required_and_named = False

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        self._maybe_force_auto_tool_parsing(request)
        reasoning, content, tool_calls = super().parse(
            model_output,
            request,
            enable_auto_tools,
            model_output_token_ids,
        )
        if tool_calls:
            from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

            # Named/required tool_choice builds FunctionCalls without
            # ID, backfill with Mistral-format IDs.
            for tc in tool_calls:
                if not tc.id:
                    tc.id = MistralToolCall.generate_random_id()
        return reasoning, content, tool_calls

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        self._maybe_force_auto_tool_parsing(request)
        return super().parse_delta(
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids,
            finished=finished,
        )
