# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.path_normalizer import normalize_pathlike_text
from vllm.reasoning.minimax_m2_reasoning_parser import MiniMaxM2ReasoningParser


class MiniMaxM2SanitizedReasoningParser(MiniMaxM2ReasoningParser):
    """MiniMax M2 reasoning parser with conservative path normalization."""

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        reasoning, content = super().extract_reasoning(model_output, request)
        return reasoning, normalize_pathlike_text(content)
