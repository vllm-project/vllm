# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses as dt
import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class Olmo3ReasoningState(enum.Enum):
    REASONING = 1
    CONTENT = 2


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

    def _tail_awaiting_marker(self) -> bool:
        # Hold only while the buffer's tail can still grow into a marker: a
        # suffix that is a proper prefix of think_start/think_end. A delta
        # that merely occurs inside a marker ("think", "in", ">") cannot
        # become one unless the text before it already ends with the right
        # characters, so holding it would only delay output.
        for marker in (self.think_start, self.think_end):
            if marker in self.buffer:
                # a completed marker in the buffer is processed, not held
                continue
            limit = min(len(self.buffer), len(marker) - 1)
            for k in range(1, limit + 1):
                if marker.startswith(self.buffer[-k:]):
                    return True
        return False

    def add_text(self, delta_text: str) -> DeltaMessage | None:
        # we start by adding the delta text to the buffer
        self.buffer += delta_text

        if self._tail_awaiting_marker():
            # the tail can still complete into a marker with the next
            # deltas, so wait for them
            return None

        # otherwise process the buffer as normal
        return self.process_buffer()

    def drain(self) -> DeltaMessage | None:
        # Flush whatever the buffer still holds once the stream is over.
        # A held tail can never complete into a marker anymore, so it is
        # emitted as-is: reasoning while the reasoning block is still open,
        # content once it closed, matching what earlier deltas streamed.
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        while len(self.buffer):
            before = len(self.buffer)
            delta_message = self.process_buffer()
            if delta_message is not None:
                if delta_message.reasoning:
                    reasoning_parts.append(delta_message.reasoning)
                if delta_message.content:
                    content_parts.append(delta_message.content)
            if len(self.buffer) == before:
                break
        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning=reasoning, content=content)


class Olmo3ReasoningParser(ReasoningParser):
    """Reasoning parser for Olmo 3 model.

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

    think_start: str = r"<think>"
    think_end: str = r"</think>"
    # </think> is split in 3 by the pre-tokenizer, first split can be tokenized
    # with an optional leading space, so there are 2 possible tokenizations
    think_end_first_split: list[str] = [r"Ġ</", r"</"]
    think_end_rest_split: list[str] = [r"think", r">"]
    # notice that the first think is optional; this allows template to
    # work in cases when we hardcode a <think> at the beginning of the
    # reasoning template.
    reasoning_regex: re.Pattern = re.compile(
        rf"^(?:{think_start})?(?P<reasoning>.*?)"
        rf"{think_end}(?P<content>.*)$",
        re.DOTALL,
    )

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.buffer = Olmo3ReasoningBuffer(
            think_start=self.think_start, think_end=self.think_end
        )
        self.think_end_first_token_ids: list[int] = [
            self.vocab[token] for token in self.think_end_first_split
        ]
        self.think_end_rest_token_ids: list[int] = [
            self.vocab[token] for token in self.think_end_rest_split
        ]

    @property
    def reasoning_start_str(self) -> str:
        return self.think_start

    @property
    def reasoning_end_str(self) -> str:
        return self.think_end

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        rest_ids = self.think_end_rest_token_ids
        rest_len = len(rest_ids)
        for i in range(len(input_ids) - rest_len, -1, -1):
            if (
                list(input_ids[i + 1 : i + 1 + rest_len]) == rest_ids
                and input_ids[i] in self.think_end_first_token_ids
            ):
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # for Olmo 3 streaming reason parsing, the stream parse
        # will call first, and the same token will be called in
        # is_reasoning_end and extract_content_ids
        # this id is not part of content, so just return [] here.
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract the reasoning content & content sections, respectively.
        If the sequence doesn't match what we expect, i.e., the model generates
        something else, all content is considered non-reasoning content.

        Args:
            model_output: Output of the model to be parsed.
            request: Request being
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
        """Extract content using token ID sequence state machine."""
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

    def finish_streaming(self) -> DeltaMessage | None:
        """Flush text the buffer still holds once the stream has ended.

        The buffer holds a delta back while it can still grow into a
        marker; when generation ends on such a delta the tail can never
        complete, and without a flush it would be dropped. Called by the
        serving layer from ``parse_delta(finished=True)``.
        """
        return self.buffer.drain()
