# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.parser.path_normalizer import (
    normalize_pathlike_text,
    normalize_tool_arguments_json,
)
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser


class MinimaxM2SanitizedToolParser(MinimaxM2ToolParser):
    """MiniMax M2 tool parser with conservative path normalization."""

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ):
        info = super().extract_tool_calls(model_output, request)
        if info.content is not None:
            info.content = normalize_pathlike_text(info.content)
        for tool_call in info.tool_calls:
            tool_call.function.arguments = (
                normalize_tool_arguments_json(tool_call.function.arguments)
                or tool_call.function.arguments
            )
        return info

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        delta = super().extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
        if delta is None:
            return None
        if delta.content is not None:
            delta.content = normalize_pathlike_text(delta.content)
        for tool_call in delta.tool_calls:
            if tool_call.function and tool_call.function.arguments is not None:
                tool_call.function.arguments = (
                    normalize_pathlike_text(tool_call.function.arguments)
                    or tool_call.function.arguments
                )
        return delta
