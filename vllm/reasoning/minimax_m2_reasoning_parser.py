# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from itertools import islice
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
)
from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)


class MiniMaxM2ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for MiniMax M2 model.

    MiniMax M2 models don't generate <think> start token, only </think> end
    token. All content before </think> is reasoning, content after is the
    actual response.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return content token IDs, stripping any re-emitted end tokens."""
        if self.end_token_id not in islice(input_ids, 0, max(0, len(input_ids) - 1)):
            return []
        after_end = input_ids[input_ids.index(self.end_token_id) + 1:]
        return [tid for tid in after_end if tid != self.end_token_id]

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if self.end_token not in model_output:
            return model_output, None
        reasoning, _, content = model_output.partition(self.end_token)
        # Strip any subsequent </think> the model re-emits in the content block
        content = content.replace(self.end_token, "")
        return reasoning, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message for streaming.

        MiniMax M2 models don't generate <think> start token, so we assume
        all content is reasoning until we encounter the </think> end token.
        """
        # Skip single end token
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.end_token_id:
            return None

        # Check if end token has already appeared in previous tokens
        # meaning we're past the reasoning phase
        if self.end_token_id in previous_token_ids:
            # Strip any re-emitted </think> tokens (model sometimes re-generates
            # the end token in a multi-token delta, bypassing the single-token guard)
            text = delta_text.replace(self.end_token, "")
            return DeltaMessage(content=text if text else None)

        # Check if end token is in delta tokens
        if self.end_token_id in delta_token_ids:
            # End token in delta, split reasoning and content
            end_index = delta_text.find(self.end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.end_token) :]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None,
                content=content if content else None,
            )

        # No end token yet, all content is reasoning
        return DeltaMessage(reasoning=delta_text)


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
