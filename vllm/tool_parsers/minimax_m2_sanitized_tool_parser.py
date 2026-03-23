# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
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
