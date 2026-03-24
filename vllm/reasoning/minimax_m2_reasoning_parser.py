# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
)
from vllm.parser.engine.registered_adapters import MinimaxM2ParserReasoningAdapter
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _count_minimax_reasoning_tokens(
    token_ids: Sequence[int], end_token_id: int | None
) -> int:
    if end_token_id is None:
        return 0

    for idx, token_id in enumerate(token_ids):
        if token_id == end_token_id:
            return idx
    return len(token_ids)


class MiniMaxM2ReasoningParser(MinimaxM2ParserReasoningAdapter):  # type: ignore[valid-type, misc]
    """
    Reasoning parser for MiniMax M2 model.

    MiniMax M2 models don't generate <think> start token, only </think> end
    token. All content before </think> is reasoning, content after is the
    actual response.
    """

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return _count_minimax_reasoning_tokens(token_ids, self.vocab.get("</think>"))


class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):
    """
    Reasoning parser for MiniMax M2 model.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.end_token_id = self.vocab.get("</think>")
        self.start_token_id = self.vocab.get("<think>")

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        end_token_id = self.end_token_id
        start_token_id = self.start_token_id
        for input_id in reversed(input_ids):
            if input_id in (end_token_id, start_token_id):
                return input_id == end_token_id
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return input_ids

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if len(previous_token_ids) == 0:
            delta_text = "<think>" + delta_text
        return DeltaMessage(content=delta_text)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        return None, "<think>" + model_output

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return _count_minimax_reasoning_tokens(token_ids, self.end_token_id)
