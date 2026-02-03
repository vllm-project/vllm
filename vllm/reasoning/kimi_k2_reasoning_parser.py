# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser


class KimiK2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Kimi K2 model.

    The Kimi K2 model uses <think>...</think> tokens to denote reasoning text,
    and may implicitly end reasoning by starting a tool call section using
    <|tool_calls_section_begin|>.
    Thinking may also begin without a </think> token.

    Kimi's thinking mode can be disabled via chat_template_kwargs.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        # Check if thinking is disabled via chat_template_kwargs
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", True))

        # If thinking is not enabled, use identity parser to fall through
        if not thinking:
            self._identity_parser = IdentityReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._identity_parser = None

        # Token definitions
        self._start_token = "<think>"
        self._end_token = "</think>"
        self._tool_section_start_token = "<|tool_calls_section_begin|>"

        # Get token IDs
        self._start_token_id = self.vocab.get(self._start_token)
        self._end_token_id = self.vocab.get(self._end_token)
        self._tool_section_start_token_id = self.vocab.get(
            self._tool_section_start_token
        )

        if self._start_token_id is None or self._end_token_id is None:
            raise RuntimeError(
                "KimiK2ReasoningParser could not locate think start/end "
                "tokens in the tokenizer!"
            )

    def _is_identity_mode(self) -> bool:
        """Check if parser is in identity mode (no reasoning extraction)."""
        return self._identity_parser is not None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        Reasoning ends when we see either:
        1. The end token (</think>)
        2. The tool section start token (<|tool_calls_section_begin|>)
        """
        if self._is_identity_mode():
            return self._identity_parser.is_reasoning_end(input_ids)

        start_token_id = self._start_token_id
        end_token_id = self._end_token_id
        tool_section_start_token_id = self._tool_section_start_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == end_token_id:
                return True
            # Implicit reasoning end via tool call section
            if (
                tool_section_start_token_id is not None
                and input_ids[i] == tool_section_start_token_id
            ):
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        """
        Check if the reasoning content ends in the input_ids on a decode step.
        """
        if self._is_identity_mode():
            return self._identity_parser.is_reasoning_end_streaming(
                input_ids, delta_ids
            )

        # Check for explicit end token or implicit tool section start in delta
        if self._end_token_id in delta_ids:
            return True
        return (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in delta_ids
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        if self._is_identity_mode():
            return self._identity_parser.extract_content_ids(input_ids)

        if self._end_token_id in input_ids:
            end_token_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._end_token_id)
            )

            if end_token_index != -1:
                return input_ids[end_token_index + 1 :]

        if (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in input_ids
        ):
            tool_section_index = (
                len(input_ids)
                - 1
                - input_ids[::-1].index(self._tool_section_start_token_id)
            )

            if tool_section_index != -1:
                return input_ids[tool_section_index:]

        # still reasoning (no content)
        return []

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        """
        if self._is_identity_mode():
            return self._identity_parser.extract_reasoning(model_output, request)

        # thinking does not require a think start token but consume it if present
        start_token_index = model_output.find(self._start_token)
        start_token_index = 0 if start_token_index != 0 else len(self._start_token)
        end_token_index = model_output.find(self._end_token)

        if end_token_index != -1:
            return (
                model_output[start_token_index:end_token_index],
                model_output[end_token_index + len(self._end_token) :] or None,
            )

        tool_section_index = model_output.find(self._tool_section_start_token)
        if tool_section_index != -1:
            return (
                model_output[start_token_index:tool_section_index],
                model_output[tool_section_index:] or None,
            )

        # still reasoning (no content)
        return (
            model_output[start_token_index:],
            None,
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
        """
        Extract reasoning content from a delta message during streaming.
        """
        if self._is_identity_mode():
            return self._identity_parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        # Skip single special tokens
        if len(delta_token_ids) == 1 and delta_token_ids[0] in [
            self._start_token_id,
            self._end_token_id,
        ]:
            return None

        if self._end_token_id in delta_token_ids:
            end_index = delta_text.find(self._end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self._end_token) :]
            return DeltaMessage(
                reasoning=reasoning, content=content if content else None
            )

        if self._tool_section_start_token_id in delta_token_ids:
            tool_index = delta_text.find(self._tool_section_start_token)
            reasoning = delta_text[:tool_index]
            content = delta_text[tool_index:]
            return DeltaMessage(reasoning=reasoning, content=content)

        # still reasoning (no end token)
        return DeltaMessage(reasoning=delta_text)
