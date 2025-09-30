# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
import dataclasses as dt
from typing import Optional, Union
import enum

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


class Olmo3ReasoningState(enum.Enum):
    IDLE = 0
    REASONING = 1
    CONTENT = 2


@dt.dataclass
class Olmo3ReasoningBuffer:
    think_start: str = "<think>"
    think_end: str = "</think>"
    text_buffer: str = ""
    state: Olmo3ReasoningState = Olmo3ReasoningState.IDLE

    @staticmethod
    def _check_string_overlap(s1: str, s2: str) -> bool:
        # fast check; containing
        if s1 in s2 or s2 in s1:
            return True

        # slower check that checks if there's any overlap between
        # prefixes and suffixes
        min_len = min(len(s1), len(s2))

        # Check if s1 suffix matches s2 prefix
        for i in range(1, min_len + 1):
            if s1[-i:] == s2[:i]:
                return True

        # Check if s2 suffix matches s1 prefix
        for i in range(1, min_len + 1):
            if s2[-i:] == s1[:i]:
                return True

        return False

    def _update_buffer_and_maybe_emit(self, delta_text: str) -> bool:
        if self._check_string_overlap(delta_text, self.think_start):
            # the delta_text **might** be a start thinking tag;
            # in doubt, we wait till the next delta_text to emit
            self.text_buffer += delta_text
            return False

        elif self._check_string_overlap(delta_text, self.think_end):
            # the delta_text **might** be a end thinking tag;
            # in doubt, we wait till the next delta_text to emit
            self.text_buffer += delta_text
            return False

        self.text_buffer += delta_text
        return True

    def _process_buffer(self) -> Optional[DeltaMessage]:
        start_think_idx = self.text_buffer.find(self.think_start)

        if start_think_idx >= 0:
            self.state = Olmo3ReasoningState.REASONING
            pretext, self.text_buffer = (
                self.text_buffer[:start_think_idx],
                self.text_buffer[start_think_idx + len(self.think_start) :],
            )
            if start_think_idx > 0:
                # this covers the case there's content before
                # the start of the reasoning block
                return DeltaMessage(content=pretext)

        end_think_idx = self.text_buffer.rfind(self.think_end)

        if end_think_idx >= 0:
            self.state = Olmo3ReasoningState.CONTENT
            pretext, self.text_buffer = (
                self.text_buffer[:end_think_idx],
                self.text_buffer[end_think_idx + len(self.think_end) :],
            )
            if end_think_idx > 0:
                # this covers the case there's content before
                # the end of the reasoning block
                return DeltaMessage(reasoning_content=pretext)

        if self.state == Olmo3ReasoningState.REASONING:
            # we are inside reasoning block, return and empty
            # the text buffer
            (
                text_buffer,
                self.text_buffer,
            ) = self.text_buffer, ""
            return DeltaMessage(reasoning_content=text_buffer)

        if self.state == Olmo3ReasoningState.CONTENT:
            # we are outside reasoning block, return and empty
            # the text buffer
            (
                text_buffer,
                self.text_buffer,
            ) = self.text_buffer, ""
            return DeltaMessage(content=text_buffer)

        # nothing to return unless we are in reasoning or content state
        return None

    def __len__(self):
        # is the length of the text buffer
        return len(self.text_buffer)

    def process(self, delta_text: str) -> Optional[DeltaMessage]:
        maybe_emit = self._update_buffer_and_maybe_emit(delta_text)

        if not maybe_emit:
            return None

        elif maybe_emit and len(self) > 0:
            return self._process_buffer()

        return None


@ReasoningParserManager.register_module("olmo3")
class Olmo3ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Olmo 3 model

    Olmo3ReasoningParser

    This class implements a reasoning parser specifically designed
    for the Olmo 33 family of models. It is responsible for parsing and
    extracting structured reasoning and answer segments from model
    outputs that follow a specific pattern.

    Key Features:
        - For non-stream output , Recognizes and extracts reasoning ("think")
         and answer ("answer") sections from text using regular expressions.
        - For stream process, it requires a token id sequences to change the
          reasoning state and other state so it maintains internal state to
          manage parsing across multiple token.

    Implementation is based on the implementation of the Hunyuan A13B Reasoning
    Parser, but with modified token ids and no <answer> token.

    # think start: "<think>": [14023, 771, *]
    # think ends: "</think>": [524, 27963, *]

    Where the value of the last token is the number of newlines.

    - No newline: 29
    - 1 newline: 397
    - 2 newlines: 1363
    - 3 newlines: 10586
    - 4 newlines: 28801
    - 5 newlines: 76819

    Six or more then it's equal to no newline.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        self.model_tokenizer = tokenizer
        self.think_start = r"<think>"
        self.think_end = r"</think>"
        # self.no_think_expr = f"{self.think_start_expr}{self.think_end_expr}"

        self.reasoning_regex = re.compile(
            rf"(?:{self.think_start}(?P<think>.*?){self.think_end})(?P<answer>.*?)$",
            re.DOTALL,
        )

        self.buffer = Olmo3ReasoningBuffer(
            think_start=self.think_start, think_end=self.think_end
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # for Olmo 3 streaming reason parsing, the stream parse
        # will call first, and the same token will be called in
        # is_reasoning_end and extract_content_ids
        # this id is not part of content, so just return [] here.
        return []

    def extract_reasoning_content(
        self,
        model_output: str,
        request: Union[ChatCompletionRequest, ResponsesRequest],
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract the reasoning content & content sections, respectively.
        If the sequence doesn't match what we expect, i.e., the model generates
        something else, all content is considered non-reasoning content.

        Args:
            model_output (str): Output of the model to be parsed.
            request (ChatCompletionRequest | ResponsesRequest): Request being processed.

        Returns:
            tuple[Optional[str], Optional[str]]: Tuple pair containing the
            reasoning content and non-reasoning content.
        """

        re_match = self.reasoning_regex.match(model_output)
        if re_match:
            think_content = re_match.group("think") or None
            answer_content = re_match.group("answer") or None
            return think_content, answer_content

        # no reasoning content
        return None, model_output

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        print()
        print("delta_text:", repr(delta_text))
        print("previous_text:", repr(previous_text))
        print("current_text:", repr(current_text))

        response = self._extract_reasoning_content_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
        print("response:", response)
        print("=======")
        return response

    def _extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """Extract content using token ID sequence state machine"""

        return self.buffer.process(delta_text)
