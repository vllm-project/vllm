# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

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
class Qwen3ReasoningParser(ReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Qwen3 model uses <think>...</think> tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.think_start_token_ids = self.model_tokenizer.encode(
            self.think_start_token, add_special_tokens=False)
        self.think_end_token_ids = self.model_tokenizer.encode(
            self.think_end_token, add_special_tokens=False)

        self.think_start_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.think_start_token_ids
        ]

        self.think_end_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.think_end_token_ids
        ]

        self.buffered_delta_text = ""

    # Very simple idea: when encountering tokens like <, think, >,
    # <, /, think, >, store them in a buffer.
    # When the last token is encountered, empty the buffer and return it.
    # If a token appears in an incorrect sequence while storing in the buffer,
    # return the preceding buffer along with the token.
    def delta_buffer(self, delta_text: str):
        # If the sequence of think_start or think_end tokens is not yet
        # complete, fill the buffer with the token and return "".
        if (delta_text in self.think_start_token_array
                or delta_text in self.think_end_token_array):
            # If delta_text is the last token of think_start_token or
            # thibk_end_token, empty the buffer and return
            # the buffered text + delta_text.
            if (delta_text == self.think_start_token_array[-1]
                    or delta_text == self.think_end_token_array[-1]):
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

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        start, _ = find_subsequence(self.think_end_token_ids, input_ids)
        return start != -1

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        _, end = find_subsequence(self.think_end_token_ids, input_ids)
        return input_ids[end:]

    def _prompt_ends_with_start_token(self,
                                      request: ChatCompletionRequest) -> bool:
        if request.vllm_xargs is not None:
            prompt = request.vllm_xargs.get("rendered_prompt", "").strip()
            return prompt.endswith(self.think_start_token)
        return False

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
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
        if (len(previous_text) >= len(self.buffered_delta_text)
                and previous_text[-len(self.buffered_delta_text):]
                == self.buffered_delta_text):
            previous_text = previous_text[:-len(self.buffered_delta_text)]

        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in self.think_start_token_ids or \
                delta_token_ids[0] in self.think_end_token_ids
        ):
            return None

        if self.think_start_token in previous_text or \
            self._prompt_ends_with_start_token(request):
            if self.think_end_token in delta_text:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content,
                                    content=content if content else None)
            elif self.think_end_token in previous_text:
                # <think> in previous, </think> in previous,
                # reasoning content continues
                return DeltaMessage(content=delta_text)
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.think_start_token in delta_text:
            if self.think_end_token in delta_text:
                # <think> in delta, </think> in delta, extract reasoning content
                start_index = delta_text.find(self.think_start_token)
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[start_index +
                                               len(self.think_start_token
                                                   ):end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content,
                                    content=content if content else None)
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        else:
            # thinking is disabled, just content
            return DeltaMessage(content=delta_text)

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """
        # Check if the model output contains the <think> and </think> tokens.
        if ((self.think_start_token not in model_output
             and not self._prompt_ends_with_start_token(request))
                or self.think_end_token not in model_output):
            return None, model_output
        # Check if the <think> is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.think_start_token)
        model_output = model_output_parts[2] if model_output_parts[
            1] else model_output_parts[0]
        # Check if the model output contains the </think> tokens.
        # If the end token is not found, return the model output as is.
        if self.think_end_token not in model_output:
            return None, model_output

        # Extract reasoning content from the model output.
        reasoning_content, _, content = model_output.partition(
            self.think_end_token)

        final_content = content or None
        return reasoning_content, final_content
