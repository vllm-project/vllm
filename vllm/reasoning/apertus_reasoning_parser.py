# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for Apertus models.

Apertus wraps its thinking between a start/end pair of special tokens. The
canonical pair is ``<|inner_prefix|>``/``<|inner_suffix|>``, but some tokenizer
builds register ``<think>``/``</think>`` at the emitted ids instead. The parser
selects whichever pair the loaded tokenizer exposes at the lower start-token id.
"""

from functools import cached_property
from typing import TYPE_CHECKING

from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Candidate (start, end) delimiter pairs, in fallback order.
_CANDIDATE_PAIRS = (
    ("<|inner_prefix|>", "<|inner_suffix|>"),
    ("<think>", "</think>"),
)


class ApertusReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for the Apertus thinking block."""

    @cached_property
    def _pair(self) -> tuple[str, str]:
        vocab = self.vocab
        present = sorted(
            (vocab[start], start, end)
            for start, end in _CANDIDATE_PAIRS
            if start in vocab and end in vocab
        )
        return (present[0][1], present[0][2]) if present else _CANDIDATE_PAIRS[0]

    @property
    def start_token(self) -> str:
        return self._pair[0]

    @property
    def end_token(self) -> str:
        return self._pair[1]

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        # With no thinking block at all (direct tool call or plain answer),
        # the base class would label the whole output as reasoning, hiding tool
        # calls from the tool parser. Return it as content instead.
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        # The base class only flips the phase once the end token appears.
        # If the start token never appeared either, this is a direct tool
        # call with no thinking block, so treat reasoning as already over.
        if self.start_token_id not in input_ids:
            return True
        return super().is_reasoning_end_streaming(input_ids, delta_ids)
