# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    FunctionCall,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.muse_glimmer_reasoning_parser import (
    MuseGlimmerReasoningParser,
)
from vllm.tool_parsers.muse_glimmer_tool_parser import MuseGlimmerToolParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MuseGlimmerParser(DelegatingParser):
    """Compose MuseGlimmer reasoning, answer, and tool channels."""

    def _muse_recipient(self, input_ids: Sequence[int]) -> str | None:
        reasoner = self._reasoning_parser
        if not isinstance(reasoner, MuseGlimmerReasoningParser):
            return None
        try:
            text = self.model_tokenizer.decode(input_ids)
        except Exception:
            return None
        return reasoner._response_recipient(text)

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        if not isinstance(self._reasoning_parser, MuseGlimmerReasoningParser):
            return super().is_reasoning_end_streaming(input_ids, delta_ids)
        recipient = self._muse_recipient(input_ids)
        return self._tool_parser is not None and recipient not in (
            None,
            "self",
            "user",
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        delta_message = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if (
            self._tool_parser is not None
            or not isinstance(self._reasoning_parser, MuseGlimmerReasoningParser)
            or delta_message is None
            or not delta_message.content
        ):
            return delta_message

        recipient = self._reasoning_parser._response_recipient(current_text)
        if recipient in (None, "self", "user"):
            return delta_message

        delta_message.content = None
        if delta_message.reasoning or delta_message.tool_calls:
            return delta_message
        return None

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

        if (
            isinstance(self._tool_parser, MuseGlimmerToolParser)
            and request.tool_choice == "none"
        ):
            tool_call_info = self.extract_tool_calls(content or "", request=request)
            return [], tool_call_info.content

        return super()._extract_tool_calls(content, request, enable_auto_tools)

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
