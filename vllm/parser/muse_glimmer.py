# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.exceptions import VLLMValidationError
from vllm.parser.abstract_parser import DelegatingParser, StreamState
from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser
from vllm.reasoning.muse_glimmer_utils import (
    advance_emitted,
    channel_seed,
    current_assistant_turn,
    flush_open_body,
    has_complete_channel,
    open_recipient,
    visible_channels,
)
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MuseGlimmerParser(DelegatingParser):
    """Compose MuseGlimmer reasoning, answer, and tool channels."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The MuseGlimmer channel framing is only handled end-to-end when the
        # muse parsers are paired: a foreign tool parser cannot read ATEM, and
        # a foreign reasoning parser reports a boundary the composite's stream
        # ownership does not expect. Reject the mix instead of leaking raw
        # channel markup to the client.
        if self._tool_parser is not None and not isinstance(
            self._tool_parser, MuseGlimmerToolParser
        ):
            raise VLLMValidationError(
                "the muse_glimmer reasoning parser only works with "
                "--tool-call-parser muse_glimmer"
            )
        if self._reasoning_parser is not None and not isinstance(
            self._reasoning_parser, MuseGlimmerReasoningParser
        ):
            raise VLLMValidationError(
                "the muse_glimmer tool parser only works with "
                "--reasoning-parser muse_glimmer"
            )

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
                    recipient = reasoner._initial_recipient
                else:
                    recipient = self._prompt_open_recipient(prompt_token_ids)
                seed = channel_seed(recipient)
                if seed is not None:
                    state.previous_text = seed
                state.prompt_reasoning_checked = True
            state.reasoning_ended = True
        return super().parse_delta(
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids=prompt_token_ids,
            finished=finished,
        )

    def _prompt_open_recipient(self, prompt_token_ids: list[int]) -> str | None:
        """Recipient of the prompt's open channel, without a muse reasoner."""
        try:
            text = self.model_tokenizer.decode(prompt_token_ids)
        except Exception:
            return None
        return open_recipient(current_assistant_turn(text))

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """Stream ownership transfers only to the paired ATEM tool parser.

        Engine-side callers (the structured-output seed in serving.py) use
        the bare reasoning parser directly, which keeps the wider boundary
        (any non-`self` channel, including `to=user`).
        """
        return isinstance(self._tool_parser, MuseGlimmerToolParser)

    def _is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        # The base implementation asks the bare reasoning parser directly,
        # whose boundary (any non-`self` channel, including `to=user`) is the
        # ENGINE-side grammar rule. The frontend phase machine must use the
        # composite's narrower rule, or a `to=user` answer ends the reasoning
        # phase mid-turn and later channels leak through the passthrough.
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

        if not has_complete_channel(state.previous_text):
            # The streaming fallback held these tails back; flush them now.
            # This also recovers text the streaming side held while a stray
            # marker (e.g. a quoted `<|start|>`) never completed a header.
            content, reasoning = flush_open_body(state.previous_text), ""
            # If the stream flipped unframed->framed without any channel ever
            # completing, framed content stayed empty and this whole-text
            # flush must resume from the pre-flip (unframed) cursor.
            content_cursor = (
                tool_parser._emitted_content_pre_flip
                if tool_parser._emitted_content_pre_flip is not None
                else tool_parser._emitted_content
            )
            if (
                "<atem:invoke" in state.previous_text
                and getattr(request, "tool_choice", None) != "none"
            ):
                # A derailed/headerless ATEM block never formed a channel:
                # salvage its complete calls at finish, like the non-streaming
                # fallback does. The streaming side parses calls only with a
                # complete channel in view, so a headerless call was never
                # emitted mid-stream; skip anything that was, all the same.
                calls = tool_parser._parse_tool_calls(
                    state.previous_text,
                    tool_parser._registered_names(request),
                )
                for i in range(tool_parser._emitted_tool_calls, len(calls)):
                    call = calls[i]
                    if delta_message is None:
                        delta_message = DeltaMessage()
                    delta_message.tool_calls = delta_message.tool_calls or []
                    delta_message.tool_calls.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=call.id,
                            function=DeltaFunctionCall(
                                name=call.function.name,
                                arguments=call.function.arguments,
                            ).model_dump(exclude_none=True),
                        )
                    )
        else:
            content, reasoning, _content_open, _reasoning_open = visible_channels(
                state.previous_text, flush_growing=True
            )
            content_cursor = tool_parser._emitted_content
        content_remainder, emitted_content = advance_emitted(content_cursor, content)
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
        # When no tool call fires (the model wrote a ``to=user`` answer), the
        # base path returns its input unchanged. That input is still raw framed
        # text -- the muse ``extract_reasoning`` deliberately preserves framing
        # for the tool parser, and with no reasoning parser nothing touched it
        # at all -- so strip the channel framing before it reaches the client.
        if (
            not tool_calls
            and out_content
            and (
                isinstance(self._reasoning_parser, MuseGlimmerReasoningParser)
                or isinstance(self._tool_parser, MuseGlimmerToolParser)
            )
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
