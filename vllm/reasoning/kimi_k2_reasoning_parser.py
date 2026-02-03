# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
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

        # Find the last occurrence of end token or tool section start
        end_index = -1

        # Look for explicit end token
        with contextlib.suppress(ValueError):
            end_index = len(input_ids) - 1 - input_ids[::-1].index(self._end_token_id)

        # Look for implicit tool section start
        if self._tool_section_start_token_id is not None:
            try:
                tool_start_index = (
                    len(input_ids)
                    - 1
                    - input_ids[::-1].index(self._tool_section_start_token_id)
                )
                if tool_start_index > end_index:
                    end_index = tool_start_index
            except ValueError:
                pass

        if end_index < 0 or end_index >= len(input_ids) - 1:
            return []

        return input_ids[end_index + 1 :]

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        """
        if self._is_identity_mode():
            return self._identity_parser.extract_reasoning(model_output, request)

        # Check if the start token is present in the model output
        if self._start_token in model_output:
            # Extract content after start token
            reasoning_content = model_output.split(self._start_token, 1)[1]

            # Check for explicit end token
            if self._end_token in reasoning_content:
                reasoning, _, content = reasoning_content.partition(self._end_token)
                return reasoning, content if content else None

            # Check for implicit tool section start
            if self._tool_section_start_token in reasoning_content:
                reasoning, _, _ = reasoning_content.partition(
                    self._tool_section_start_token
                )
                # Content is empty when tool section starts implicitly
                return reasoning, None

            # Neither end token nor tool section found - all reasoning
            return reasoning_content, None

        # No start token found - check if there's implicit tool section
        if self._tool_section_start_token in model_output:
            reasoning, _, _ = model_output.partition(self._tool_section_start_token)
            # Content is empty when tool section starts implicitly
            return reasoning if reasoning else None, None

        # No reasoning markers found - default to reasoning mode
        return model_output, None

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
            self._tool_section_start_token_id,
        ]:
            return None

        start_in_prev = self._start_token_id in previous_token_ids
        start_in_delta = self._start_token_id in delta_token_ids
        end_in_prev = self._end_token_id in previous_token_ids
        end_in_delta = self._end_token_id in delta_token_ids
        tool_in_delta = (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in delta_token_ids
        )

        # Case 1: Start token was already seen
        if start_in_prev:
            if end_in_delta:
                # Explicit end found in delta
                end_index = delta_text.find(self._end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self._end_token) :]
                return DeltaMessage(
                    reasoning=reasoning, content=content if content else None
                )
            elif tool_in_delta:
                # Implicit end via tool section
                tool_index = delta_text.find(self._tool_section_start_token)
                reasoning = delta_text[:tool_index]
                # Don't include content - tool parser will handle it
                return DeltaMessage(reasoning=reasoning, content=None)
            elif end_in_prev:
                # Already past reasoning
                return DeltaMessage(content=delta_text)
            else:
                # Still in reasoning
                return DeltaMessage(reasoning=delta_text)

        # Case 2: Start token in current delta
        elif start_in_delta:
            if end_in_delta:
                # Both start and end in same delta
                start_index = delta_text.find(self._start_token)
                end_index = delta_text.find(self._end_token)
                reasoning = delta_text[start_index + len(self._start_token) : end_index]
                content = delta_text[end_index + len(self._end_token) :]
                return DeltaMessage(
                    reasoning=reasoning, content=content if content else None
                )
            elif tool_in_delta:
                # Start and tool section in same delta - unlikely but handle it
                start_index = delta_text.find(self._start_token)
                tool_index = delta_text.find(self._tool_section_start_token)
                reasoning = delta_text[
                    start_index + len(self._start_token) : tool_index
                ]
                return DeltaMessage(reasoning=reasoning, content=None)
            else:
                # Only start in delta - beginning of reasoning
                start_index = delta_text.find(self._start_token)
                reasoning = delta_text[start_index + len(self._start_token) :]
                return DeltaMessage(reasoning=reasoning)

        # Case 3: No start token seen yet - default to reasoning mode
        else:
            # Check if tool section starts without explicit reasoning markers
            if tool_in_delta:
                # Tool section starts implicitly - any text before it is reasoning
                tool_index = delta_text.find(self._tool_section_start_token)
                if tool_index > 0:
                    reasoning = delta_text[:tool_index]
                    return DeltaMessage(reasoning=reasoning, content=None)
                else:
                    # Tool section at start - no reasoning to extract
                    return None
            return DeltaMessage(reasoning=delta_text)
