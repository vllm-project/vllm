# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses as dt
import enum
from collections.abc import Sequence
from typing import Optional, Union

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage, ResponsesRequest)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


class Olmo3ReasoningState(enum.Enum):
    IDLE = 0
    REASONING = 1
    CONTENT = 2


@dt.dataclass(frozen=True)
class Indices:
    start: int
    end: int

    def __len__(self):
        return self.end - self.start


def string_overlap(a: str,
                   b: str) -> tuple[Optional[Indices], Optional[Indices]]:
    """
    Find the longest overlap where the end of string a matches the start
    of string b.

    Args:
        a: First string
        b: Second string

    Returns:
        Tuple of IndicesTuples representing the overlapping portions in each
        string, or a tuple of None if no overlap exists
    """

    # swap so a is always the shorter string
    a, b, swap = (a, b, False) if len(a) < len(b) else (b, a, True)

    # first check: is a fully contained in b?
    if a in b:
        ind_a = Indices(0, len(a))
        ind_b = Indices(b.index(a), b.index(a) + len(a))
        return (ind_b, ind_a) if swap else (ind_a, ind_b)

    # second check: does the end of a overlap with the
    #               beginning of b?
    for i in range(len(a) - 1, 1, -1):
        if a[-i:] == b[:i]:
            ind_a = Indices(len(a) - i, len(a))
            ind_b = Indices(0, i)
            return (ind_b, ind_a) if swap else (ind_a, ind_b)

    # third check: does the beginning of a overlap with
    #              the end of b?
    for i in range(len(a) - 1, 1, -1):
        if b[-i:] == a[:i]:
            ind_a = Indices(0, i)
            ind_b = Indices(len(b) - i, len(b))
            return (ind_b, ind_a) if swap else (ind_a, ind_b)

    return None, None


@dt.dataclass
class Olmo3ReasoningBuffer:
    think_start: str = "<think>"
    think_end: str = "</think>"
    text_buffer: str = ""
    state: Olmo3ReasoningState = Olmo3ReasoningState.IDLE

    def process_buffer(self) -> Optional[DeltaMessage]:
        start_think_idx = self.text_buffer.find(self.think_start)

        if start_think_idx >= 0:
            self.state = Olmo3ReasoningState.REASONING
            pretext, self.text_buffer = (
                self.text_buffer[:start_think_idx],
                self.text_buffer[start_think_idx + len(self.think_start):],
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
                self.text_buffer[end_think_idx + len(self.think_end):],
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

    def add_text(self, delta_text: str) -> Optional[DeltaMessage]:
        # we start by adding the delta text to the buffer
        self.text_buffer += delta_text

        # setting this to empty before starting
        delta_message: Optional[DeltaMessage] = None

        # we start by computing the overlap between the delta_text
        # and start/end of think tokens.
        _, overlap_think_start = string_overlap(delta_text, self.think_start)
        _, overlap_think_end = string_overlap(delta_text, self.think_end)

        partial_overlap_start = overlap_think_start is not None and len(
            overlap_think_start) < len(self.think_start)
        partial_overlap_end = overlap_think_end is not None and len(
            overlap_think_end) < len(self.think_end)

        if (partial_overlap_start and self.think_start in self.text_buffer
                and not partial_overlap_end):
            # we can only process the buffer if partial overlap
            # is the last part of think token (thus causing
            # text_buffer to contain the start of think token)
            # and there are no partial overlaps with end think
            delta_message = self.process_buffer()

        elif partial_overlap_end and self.think_end in self.text_buffer:
            # same as before (partial overlap only allowed)
            # if the buffer contains the end think token,
            # but we don't have to check for partial overlap
            # with start think token because they are handled
            # by the previous condition
            delta_message = self.process_buffer()

        elif partial_overlap_start or partial_overlap_end:
            # in general, if there are overlaps, we don't
            # process the buffer because we want to wait until
            # the think token is fully completed.
            return None
        else:
            # we process the buffer as normal
            delta_message = self.process_buffer()

        # one final throughput improvement; sometimes the delta message
        # is None, because process_buffer() processed some of the buffer
        # w/o emitting a message (e.g., `<think></think>` -> `</think>`)
        # in this case, we keep advancing till delta message is not None
        # or the text buffer is empty
        while delta_message is None and len(self) > 0:
            delta_message = self.process_buffer()

        return delta_message


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

        self.buffer = Olmo3ReasoningBuffer(think_start=self.think_start,
                                           think_end=self.think_end)

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
            request (ChatCompletionRequest | ResponsesRequest): Request being
                processed.

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
        """Extract content using token ID sequence state machine"""

        delta_message = self.buffer.add_text(delta_text)
        if (delta_message is None
                and self.buffer.think_end in self.buffer.text_buffer):
            # this is a bit hacky, but, because of how the buffer is
            # constructed, if the last delta_text contains characters that
            # marks the end of thinking tokens, then messages in the buffer
            # would never be processed because we get no other turn.to get
            # around that, we check if the text buffer contains the end of
            # thinking tokens, and, if so, we reprocess the buffer again.
            delta_message = self.buffer.process_buffer()

        return delta_message
