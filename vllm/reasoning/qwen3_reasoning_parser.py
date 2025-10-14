# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

logger = init_logger(__name__)


def find_subsequence(sublist, mainlist):
    s_len = len(sublist)
    m_len = len(mainlist)
    if s_len > m_len:
        return -1, -1
    for i in range(m_len - s_len + 1):
        for j, s in enumerate(sublist):
            if mainlist[i + j] != s:
                break
        else:
            return i, i + s_len
    return -1, -1


@ReasoningParserManager.register_module("qwen3")
class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.
    """

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, args, kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self.think_start_token_ids = self.model_tokenizer.encode(
            self.start_token, add_special_tokens=False
        )
        self.think_end_token_ids = self.model_tokenizer.encode(
            self.end_token, add_special_tokens=False
        )

        self.think_start_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.think_start_token_ids
        ]

        self.think_end_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.think_end_token_ids
        ]

        self.buffered_delta_text = ""
        self.cached_prompt_tokens = []
        self.cached_prompt = ""
        self.cached_outputs = []

    # Very simple idea: when encountering tokens like <, think, >,
    # <, /, think, >, store them in a buffer.
    # When the last token is encountered, empty the buffer and return it.
    # If a token appears in an incorrect sequence while storing in the buffer,
    # return the preceding buffer along with the token.
    def delta_buffer(self, delta_text: str):
        # If the sequence of think_start or think_end tokens is not yet
        # complete, fill the buffer with the token and return "".
        if (
            delta_text in self.think_start_token_array
            or delta_text in self.think_end_token_array
        ):
            # If delta_text is the last token of start_token or
            # thibk_end_token, empty the buffer and return
            # the buffered text + delta_text.
            if (
                delta_text == self.think_start_token_array[-1]
                or delta_text == self.think_end_token_array[-1]
            ):
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                self.buffered_delta_text = self.buffered_delta_text + delta_text
                return ""
        else:
            if self.buffered_delta_text:
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                return delta_text

    def _get_decoded(self, input_ids: list[int], is_prompt: bool) -> str:
        if is_prompt:
            self.cached_prompt_tokens = input_ids
            if not self.cached_prompt:
                self.cached_prompt = self.model_tokenizer.decode(input_ids)
            return self.cached_prompt
        self.cached_outputs.extend(input_ids)
        return self.model_tokenizer.decode(self.cached_outputs)

    def is_reasoning_end(self, input_ids: list[int], is_prompt: bool) -> bool:
        end_idx, _ = find_subsequence(self.think_end_token_ids, input_ids)
        if end_idx != -1:
            inputs = ""
            thinking_disabled = True
        else:
            inputs = self._get_decoded(input_ids, is_prompt)
            thinking_disabled = self.end_token in inputs

        if is_prompt:
            start_idx, _ = find_subsequence(self.think_start_token_ids, input_ids)
            if start_idx != -1:
                thinking_enabled = True
            else:
                inputs = inputs or self._get_decoded(input_ids, is_prompt)
                thinking_enabled = self.start_token in inputs

            if thinking_enabled:
                return thinking_disabled
            return True
        return thinking_disabled

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        _, end = find_subsequence(self.think_end_token_ids, input_ids)
        if end != -1:
            return input_ids[end:]
        else:
            full_content = self.model_tokenizer.decode(input_ids)
            parts = full_content.partition(self.end_token)
            content = parts[2] if parts[2] else parts[0]
            return self.model_tokenizer.encode(content, add_special_tokens=False)

    def _prompt_contains_start_token(self) -> bool:
        if not self.cached_prompt_tokens:
            return False

        start_idx, _ = find_subsequence(
            self.think_start_token_ids, self.cached_prompt_tokens
        )
        if start_idx != -1:
            return True
        inputs = self._get_decoded(self.cached_prompt_tokens, True)
        return self.start_token in inputs

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content
        """

        delta_text = self.delta_buffer(delta_text)
        # If the last characters of previous_text
        # match self.buffered_delta_text, remove only the matching part.
        if (
            len(previous_text) >= len(self.buffered_delta_text)
            and previous_text[-len(self.buffered_delta_text) :]
            == self.buffered_delta_text
        ):
            previous_text = previous_text[: -len(self.buffered_delta_text)]

        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in self.think_start_token_ids
            or delta_token_ids[0] in self.think_end_token_ids
        ):
            return None

        if self.start_token in previous_text or self._prompt_contains_start_token():
            if self.end_token in delta_text:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            elif self.end_token in previous_text:
                # <think> in previous, </think> in previous,
                # reasoning content continues
                return DeltaMessage(content=delta_text)
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.start_token in delta_text:
            if self.end_token in delta_text:
                # <think> in delta, </think> in delta, extract reasoning content
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[
                    start_index + len(self.start_token) : end_index
                ]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning_content=reasoning_content,
                    content=content if content else None,
                )
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        else:
            # thinking is disabled, just content
            return DeltaMessage(content=delta_text)

    def extract_reasoning_content(
        self,
        model_output: str,
        prompt_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        Qwen3 has stricter requirements - it needs both start and end tokens
        to be present, unlike other models that work with just the end token.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        self.cached_prompt_tokens = prompt_token_ids

        # Check if the model output contains both <think> and </think> tokens.
        if self.start_token not in model_output or self.end_token not in model_output:
            return None, model_output

        # Check if the <think> is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # Check if the model output contains the </think> tokens.
        # If the end token is not found, return the model output as is.
        if self.end_token not in model_output:
            return None, model_output

        # Extract reasoning content from the model output.
        reasoning_content, _, content = model_output.partition(self.end_token)

        final_content = content or None
        return reasoning_content, final_content
