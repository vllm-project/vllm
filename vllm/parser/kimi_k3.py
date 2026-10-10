# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    FunctionCall,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.kimi_k3_reasoning_parser import KimiK3ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class KimiK3Parser(DelegatingParser):
    """Compose the Kimi K3 reasoning and tool parsers for XTML output."""

    # TODO: Switch Kimi K3 to the parser engine once its XTML reasoning/tool
    # path is covered there.
    def _extract_tool_calls(
        self,
        content: str | None,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
    ) -> tuple[list[FunctionCall] | None, str | None]:
        if self._tool_parser is None or not enable_auto_tools:
            return super()._extract_tool_calls(content, request, enable_auto_tools)

        tool_call_info = self.extract_tool_calls(content or "", request=request)
        if request.tool_choice == "none":
            return [], tool_call_info.content
        if not tool_call_info.tools_called:
            return None, tool_call_info.content

        tool_calls = [
            FunctionCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in tool_call_info.tool_calls
        ]
        parsed_content = tool_call_info.content
        if parsed_content and parsed_content.strip() == "":
            parsed_content = None
        return tool_calls, parsed_content

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
        if request.tool_choice != "none":
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

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        state = self._stream_state
        previous_content = state.previous_text if state.reasoning_ended else ""
        delta_message = super().parse_delta(
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids,
            finished=finished,
        )

        if (
            self._tool_parser is not None
            or not isinstance(self._reasoning_parser, KimiK3ReasoningParser)
            or not state.reasoning_ended
            or delta_message is None
        ):
            return delta_message

        stripped = self._reasoning_parser.strip_content_streaming(
            previous_text=previous_content,
            current_text=state.previous_text,
        )
        delta_message.content = stripped.content if stripped is not None else None
        if (
            delta_message.role is None
            and delta_message.content is None
            and delta_message.reasoning is None
            and not delta_message.tool_calls
        ):
            return None
        return delta_message
