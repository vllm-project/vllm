# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for Apertus models.

Apertus wraps its thinking between ``<|inner_prefix|>`` and ``<|inner_suffix|>``.
The tokenizer also rewrites ``<think>``/``</think>`` to that pair, but only
through its ``normalizer``, which runs on the encode path; the emitted ids always
detokenize to ``<|inner_*|>``, so only that pair delimits generated reasoning.
"""

from typing import TYPE_CHECKING

from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class ApertusReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for the Apertus thinking block."""

    @property
    def start_token(self) -> str:
        return "<|inner_prefix|>"

    @property
    def end_token(self) -> str:
        return "<|inner_suffix|>"

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        # With no thinking block at all (direct tool call or plain answer),
        # the base class would label the whole output as reasoning, hiding tool
        # calls from the tool parser. Return it as content instead.
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output, request)
