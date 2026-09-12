# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import DeltaMessage, FunctionCall
from vllm.exceptions import VLLMValidationError
from vllm.parser.abstract_parser import DelegatingParser, StreamState
from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser
from vllm.reasoning.muse_glimmer_utils import (
    advance_emitted,
    current_assistant_turn,
    open_recipient,
    safe_open_body,
    visible_channels,
)
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MuseGlimmerParser(DelegatingParser):
    """Compose MuseGlimmer reasoning, answer, and tool channels."""

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Reject caller output constraints that collide with ATEM tools."""
        if (
            self._tool_parser is not None
            and request.tools
            and request.tool_choice != "none"
        ):
            constraint = None
            if request.structured_outputs is not None:
                constraint = "structured_outputs"
            elif (
                response_format := getattr(request, "response_format", None)
            ) is not None and getattr(response_format, "type", None) != "text":
                constraint = "response_format"
            elif (
                (text := getattr(request, "text", None)) is not None
                and (fmt := getattr(text, "format", None)) is not None
                and getattr(fmt, "type", None) != "text"
            ):
                constraint = "text.format"
            if constraint is not None:
                raise VLLMValidationError(
                    "MuseGlimmer tool calling cannot be combined with "
                    "response_format, text.format, or structured_outputs.",
                    parameter=constraint,
                )
        return super().adjust_request(request)

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        """Let the ATEM segmenter own streams paired with its tool parser."""
        state = self._stream_state
        if isinstance(self._tool_parser, MuseGlimmerToolParser):
            if not state.prompt_reasoning_checked and prompt_token_ids is not None:
                reasoner = self._reasoning_parser
                if isinstance(reasoner, MuseGlimmerReasoningParser):
                    reasoner.adjust_initial_state_from_prompt(prompt_token_ids)
                    if reasoner._initial_recipient is not None:
                        state.previous_text = (
                            f"to={reasoner._initial_recipient}<|message|>"
                        )
            state.reasoning_ended = True
            state.prompt_reasoning_checked = True
        return super().parse_delta(
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids=prompt_token_ids,
            finished=finished,
        )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """Bypass frontend reasoning only for the paired ATEM tool parser."""
        if isinstance(self._tool_parser, MuseGlimmerToolParser):
            return True
        if isinstance(self._reasoning_parser, MuseGlimmerReasoningParser):
            try:
                text = self.model_tokenizer.decode(input_ids)
            except Exception:
                return False
            recipient = open_recipient(current_assistant_turn(text))
            return self._tool_parser is not None and recipient not in (
                None,
                "self",
                "user",
            )
        return super().is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return self.is_reasoning_end(input_ids)

    def finalize_generation(
        self,
        delta_message: DeltaMessage | None,
        request: ChatCompletionRequest | ResponsesRequest,
        state: StreamState,
    ) -> DeltaMessage | None:
        delta_message = super().finalize_generation(delta_message, request, state)
        reasoner = self._reasoning_parser
        tool_parser = self._tool_parser

        if isinstance(reasoner, MuseGlimmerReasoningParser) and not isinstance(
            tool_parser, MuseGlimmerToolParser
        ):
            reasoning_remainder = reasoner.get_streaming_fallback_reasoning(
                state.previous_text
            )
            if reasoning_remainder:
                if delta_message is None:
                    delta_message = DeltaMessage()
                delta_message.reasoning = (
                    delta_message.reasoning or ""
                ) + reasoning_remainder

        if not isinstance(tool_parser, MuseGlimmerToolParser):
            return delta_message

        content, reasoning, content_open, _reasoning_open = visible_channels(
            state.previous_text
        )
        if content_open:
            content = safe_open_body(content)
        content_remainder, emitted_content = advance_emitted(
            tool_parser._emitted_content, content
        )
        reasoning_remainder, emitted_reasoning = advance_emitted(
            tool_parser._emitted_reasoning, reasoning
        )
        if not content_remainder and not reasoning_remainder:
            return delta_message

        if delta_message is None:
            delta_message = DeltaMessage()
        if content_remainder:
            delta_message.content = (delta_message.content or "") + content_remainder
            tool_parser._emitted_content = emitted_content
        if reasoning_remainder:
            delta_message.reasoning = (
                delta_message.reasoning or ""
            ) + reasoning_remainder
            tool_parser._emitted_reasoning = emitted_reasoning
        return delta_message

    def _extract_tool_calls(
        self,
        content: str | None,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
    ) -> tuple[list[FunctionCall] | None, str | None]:
        if self._tool_parser is None and isinstance(
            self._reasoning_parser, MuseGlimmerReasoningParser
        ):
            return [], MuseGlimmerToolParser._extract_content(content or "")

        tool_calls, out_content = super()._extract_tool_calls(
            content, request, enable_auto_tools
        )
        # ``extract_reasoning`` returns the raw framed turn as ``content`` so the
        # tool parser can segment it. When no tool call fires (the model wrote a
        # ``to=user`` answer), the base path returns that raw input unchanged, so
        # strip the channel framing before it reaches the client.
        if (
            not tool_calls
            and out_content
            and isinstance(self._reasoning_parser, MuseGlimmerReasoningParser)
        ):
            out_content = MuseGlimmerToolParser._extract_content(out_content)
        return tool_calls, out_content

    def _extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest | ResponsesRequest,
        tool_call_idx: int | None = None,
        tool_call_id_type: str = "random",
        function_name_returned: bool = False,
    ) -> tuple[DeltaMessage | None, bool]:
        if not (
            isinstance(self._tool_parser, MuseGlimmerToolParser)
            and request.tool_choice == "none"
        ):
            return super()._extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                request,
                tool_call_idx=tool_call_idx,
                tool_call_id_type=tool_call_id_type,
                function_name_returned=function_name_returned,
            )

        delta_message = self.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
        if delta_message is not None:
            delta_message.tool_calls = []
        return delta_message, False
