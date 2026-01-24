# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses as dt
import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)


class Olmo3ReasoningState(enum.Enum):
    REASONING = 1
    CONTENT = 2


@dt.dataclass(frozen=True)
class Indices:
    start: int
    end: int

    def __len__(self):
        return self.end - self.start


def string_overlap(a: str, b: str) -> tuple[Indices | None, Indices | None]:
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
    for i in range(len(a) - 1, 0, -1):
        if a[-i:] == b[:i]:
            ind_a = Indices(len(a) - i, len(a))
            ind_b = Indices(0, i)
            return (ind_b, ind_a) if swap else (ind_a, ind_b)

    # third check: does the beginning of a overlap with
    #              the end of b?
    for i in range(len(a) - 1, 0, -1):
        if b[-i:] == a[:i]:
            ind_a = Indices(0, i)
            ind_b = Indices(len(b) - i, len(b))
            return (ind_b, ind_a) if swap else (ind_a, ind_b)

    return None, None


@dt.dataclass
class Olmo3ReasoningBuffer:
    think_start: str = "<think>"
    think_end: str = "</think>"
    buffer: str = ""

    # we start in reasoning state to support cases where we hardcode
    # <think> as the start of the reasoning block.
    # In those cases, the only token we will see is </think>, which
    # is when we switch to content state.
    state: Olmo3ReasoningState = Olmo3ReasoningState.REASONING

    def process_buffer(self) -> DeltaMessage | None:
        start_think_idx = self.buffer.find(self.think_start)

        if start_think_idx >= 0:
            self.state = Olmo3ReasoningState.REASONING
            pretext, self.buffer = (
                self.buffer[:start_think_idx],
                self.buffer[start_think_idx + len(self.think_start) :],
            )
            if start_think_idx > 0:
                # this covers the case there's content before
                # the start of the reasoning block
                return DeltaMessage(content=pretext)

        end_think_idx = self.buffer.rfind(self.think_end)

        if end_think_idx >= 0:
            self.state = Olmo3ReasoningState.CONTENT
            pretext, self.buffer = (
                self.buffer[:end_think_idx],
                self.buffer[end_think_idx + len(self.think_end) :],
            )
            if end_think_idx > 0:
                # this covers the case there's content before
                # the end of the reasoning block
                return DeltaMessage(reasoning=pretext)

        if self.state == Olmo3ReasoningState.REASONING:
            # we are inside reasoning block, return and empty
            # the text buffer
            (
                text_buffer,
                self.buffer,
            ) = self.buffer, ""
            return DeltaMessage(reasoning=text_buffer)

        if self.state == Olmo3ReasoningState.CONTENT:
            # we are outside reasoning block, return and empty
            # the text buffer
            (
                text_buffer,
                self.buffer,
            ) = self.buffer, ""
            return DeltaMessage(content=text_buffer)

        # nothing to return unless we are in reasoning or content state
        return None

    def __len__(self):
        # is the length of the text buffer
        return len(self.buffer)

    def add_text(self, delta_text: str) -> DeltaMessage | None:
        # we start by adding the delta text to the buffer
        self.buffer += delta_text

        # setting this to empty before starting
        delta_message: DeltaMessage | None = None

        # we start by computing the overlap between the delta_text
        # and start/end of think tokens.
        _, overlap_think_start = string_overlap(delta_text, self.think_start)
        _, overlap_think_end = string_overlap(delta_text, self.think_end)

        partial_overlap_start = overlap_think_start is not None and len(
            overlap_think_start
        ) < len(self.think_start)
        partial_overlap_end = overlap_think_end is not None and len(
            overlap_think_end
        ) < len(self.think_end)

        if (
            partial_overlap_start
            and self.think_start in self.buffer
            and not partial_overlap_end
        ):
            # we can only process the buffer if partial overlap
            # is the last part of think token (thus causing
            # text_buffer to contain the start of think token)
            # and there are no partial overlaps with end think
            delta_message = self.process_buffer()

        elif partial_overlap_end and self.think_end in self.buffer:
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

        return delta_message


class Olmo3ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Olmo 3 model

    Olmo3ReasoningParser

    This class implements a reasoning parser specifically designed for the
    Olmo 3 family of models. Olmo 3 models do not use special tokens to
    indicate reasoning; rather, reasoning trace is wrapped in `<think>` and
    `</think>`, which are tokenized using standard vocabulary entries.
    Because of this, the parser operates in string space, accumulating the
    characters in a buffer until it sees `<think>` or `</think>`. tokens
    to switch modes.

    Key Features:
        - For non-stream output, Recognizes and extracts reasoning (text
          bracketed by `<think>` and `</think>`) and content (everything
          after the first `</think>`).
        - For stream process, it uses a buffer to accumulate delta text,
          and output progressive delta messages as soon as thinking starts
          or ends.
        - For reliability, some Olmo 3 models may hardcode the first
          `<think>` token is the input text (similar to Deepseek R1,
          or reasoning-only Qwen models). To support such variants, the
          parser can optionally work in cases where the first `<think>`
          token is missing from generation.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        self.think_start = r"<think>"
        self.think_end = r"</think>"

        # notice that the first think is optional; this allows template to
        # work in cases when we hardcode a <think> at the beginning of the
        # reasoning template.
        reasoning_expr = (
            rf"^(?:{self.think_start})?(?P<reasoning>.*?)"
            + rf"{self.think_end}(?P<content>.*)$"
        )
        self.reasoning_regex = re.compile(reasoning_expr, re.DOTALL)

        self.buffer = Olmo3ReasoningBuffer(
            think_start=self.think_start, think_end=self.think_end
        )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        text = self.model_tokenizer.decode(input_ids)
        return self.think_end in text

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # for Olmo 3 streaming reason parsing, the stream parse
        # will call first, and the same token will be called in
        # is_reasoning_end and extract_content_ids
        # this id is not part of content, so just return [] here.
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
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
            reasoning = re_match.group("reasoning") or None
            content = re_match.group("content") or None
            return reasoning, content

        # no reasoning content
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract content using token ID sequence state machine"""

        delta_message = self.buffer.add_text(delta_text)
        if delta_message is None and self.buffer.think_end in self.buffer.buffer:
            # this is a bit hacky, but, because of how the buffer is
            # constructed, if the last delta_text contains characters that
            # marks the end of thinking tokens, then messages in the buffer
            # would never be processed because we get no other turn. To get
            # around that, we check if the text buffer contains the end of
            # thinking tokens, and, if so, we reprocess the buffer again.
            delta_message = self.buffer.process_buffer()

        return delta_message
