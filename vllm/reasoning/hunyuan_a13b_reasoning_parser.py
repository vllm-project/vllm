# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class HunyuanA13BReasoningParser(ReasoningParser):
    """Reasoning parser for Hunyuan A13B Model.

    HunyuanReasoningParser

    This class implements a reasoning parser specifically designed
    for the Hunyuan A13B Model. It is responsible for parsing and
    extracting structured reasoning and answer segments from model
    outputs that follow a specific pattern.

    Key Features:
        - For non-stream output , Recognizes and extracts reasoning ("think")
         and answer ("answer") sections from text using regular expressions.
        - For stream process, it requires a token id sequences to change the
          reasoning state and other state so it maintains internal state to
          manage parsing across multiple token.


    think start: "<think>\n": [14023, 771, 397]
    think ends: "\n</think>\n<answer>\n": [198, 524, 27963, 397, 27, 9399, 397]
    response ends: "\n</answer>": [524, 9399, 29]
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.think_start_expr = r"<think>\n"
        self.think_end_expr = r"\n</think>\n"

        self.response_start_expr = r"\n</think>\n<answer>\n"
        self.response_end_expr = r"\n</answer>"

        self.full_match_reasoning_regex = re.compile(
            rf"(?:{self.think_start_expr}(.*?){self.response_start_expr})?(.*?){self.response_end_expr}",
            re.DOTALL,
        )

        self.half_match_reasoning_regex = re.compile(
            rf"{self.think_start_expr}(.*?){self.response_start_expr}(.*)", re.DOTALL
        )

        self.think_start_ids = [14023, 771, 397]
        self.think_start_ids_fast = [14023, 771, 1363]
        self.response_start_ids = [198, 524, 27963, 397, 27, 9399, 397]
        self.response_start_ids_fast = [524, 27963, 397, 27, 9399, 397]
        self.response_end_ids = [198, 524, 9399, 29]
        self.fast_think_ids = [14023, 771, 1363, 524, 27963, 397, 27, 9399, 397]

        # when state change, send out all the buffered text in last state
        self.buffered_text: list[str] = []
        self.buffered_ids: list[int] = []

        self.current_state = "reasoning"
        self.all_states = ["reasoning", "response"]

        self.current_state = "idle"
        self.expected_sequence = self.think_start_ids
        # this sequence only for the think start, it has two way to start.
        self.expected_sequence_side = self.think_start_ids_fast
        self.sequence_index = 0
        self.token_buffer: list[int] = []
        self.text_buffer = ""

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self.current_state == "response"

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # for hunyuan streaming reason parsing, the stream parse
        # will call first, and the same token will be called in
        # is_reasoning_end and extract_content_ids
        # this id is not part of content, so just return [] here.
        return []

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """Extract the reasoning content & content sections, respectively.
        If the sequence doesn't match what we expect, i.e., the model generates
        something else, all content is considered non-reasoning content.

        Args:
            model_output (str): Output of the model to be parsed.
            request (ChatCompletionRequest): Request being processed.

        Returns:
            tuple[Optional[str], Optional[str]]: Tuple pair containing the
            reasoning content and non-reasoning content.

        """
        re_match = self.full_match_reasoning_regex.findall(model_output)
        if re_match:
            reasoning, response_content = re_match[0]
            if len(reasoning) == 0:
                reasoning = None
            if len(response_content) == 0:
                response_content = None
            return reasoning, response_content

        fallback_regex = self.half_match_reasoning_regex
        fallback_match = fallback_regex.findall(model_output)
        if fallback_match:
            reasoning, response_content = fallback_match[0]

            if response_content.endswith(self.response_end_expr):
                response_content = response_content[: -len(self.response_end_expr)]

            if len(reasoning) == 0:
                reasoning = None
            if len(response_content) == 0:
                response_content = None

            return reasoning, response_content

        return None, model_output

    def _iter_delta_tokens(
        self,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        delta_text: str,
    ) -> Iterator[tuple[int, str]]:
        """Yield each token of a streaming delta together with its own text.

        The engine reports one generation step per call and a step can carry
        several tokens (speculative decoding, MTP), while the state machine below
        advances one token at a time. ``delta_text`` describes the whole step, so
        per-token text is decoded from the token ids when the delta is longer than
        one token. The single-token case keeps using the engine's ``delta_text``.
        """
        if len(delta_token_ids) == 1:
            yield delta_token_ids[0], delta_text
            return

        prev_len = len(previous_token_ids)
        # Text already emitted for this stream. A decode can end in U+FFFD when a
        # character spans several tokens (byte-fallback tokenization); that
        # replacement character is superseded once the character completes, which
        # would break the prefix relation the slice below relies on. Dropping the
        # incomplete tail keeps the emitted text a real prefix of later decodes.
        emitted = self.model_tokenizer.decode(
            list(current_token_ids[:prev_len]), skip_special_tokens=False
        ).rstrip("\ufffd")
        for offset, token in enumerate(delta_token_ids):
            cur_text = self.model_tokenizer.decode(
                list(current_token_ids[: prev_len + offset + 1]),
                skip_special_tokens=False,
            )
            if cur_text.endswith("\ufffd"):
                # Unfinished byte sequence: the token that completes the character
                # emits its text. Same guard as detokenize_incrementally.
                yield token, ""
                continue
            yield token, cur_text[len(emitted) :]
            emitted = cur_text

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
        # Define sequences
        think_start_sequence = self.think_start_ids
        response_start_sequence = self.response_start_ids
        response_end_sequence = self.response_end_ids

        if not delta_token_ids:
            return None

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        def check_token_with_sequence(token):
            if self.current_state == "idle" or self.current_state == "think":
                return (
                    token == self.expected_sequence[self.sequence_index]
                    or token == self.expected_sequence_side[self.sequence_index]
                )
            else:
                return token == self.expected_sequence[self.sequence_index]

        def check_last_token(token):
            if self.current_state == "idle" or self.current_state == "think":
                # only return true if it's judge using a side sequence.
                if (
                    self.sequence_index - 1 < len(self.expected_sequence_side)
                    and token == self.expected_sequence_side[self.sequence_index - 1]
                ):
                    return self.sequence_index == len(self.expected_sequence_side)
                else:
                    return self.sequence_index == len(self.expected_sequence)
            else:
                return self.sequence_index == len(self.expected_sequence)

        for token, token_text in self._iter_delta_tokens(
            previous_token_ids, current_token_ids, delta_token_ids, delta_text
        ):
            # Check if token matches expected sequence
            token_in_state_seq = check_token_with_sequence(token)

            if token_in_state_seq:
                # Store matching token
                self.token_buffer.append(token)
                self.text_buffer += token_text
                self.sequence_index += 1
                ## state change from idle->think->response->idle

                # Check if sequence fully matched
                if check_last_token(token):
                    # State transition
                    if self.current_state == "idle":
                        self.current_state = "think"
                        self.expected_sequence = response_start_sequence
                        self.expected_sequence_side = self.response_start_ids_fast
                    elif self.current_state == "think":
                        self.current_state = "response"
                        self.expected_sequence = response_end_sequence
                    elif self.current_state == "response":
                        self.current_state = "idle"
                        self.expected_sequence = think_start_sequence
                        self.expected_sequence_side = self.think_start_ids_fast

                    # Reset matching state
                    self.sequence_index = 0
                    self.token_buffer = []
                    self.text_buffer = ""
                    # Do not send content for state transition texts.
            else:
                # Sequence broken - handle buffered content
                if self.token_buffer and len(self.token_buffer) > 0:
                    # Send buffered tokens
                    buffered_content = self.text_buffer + token_text
                    # Reset matching state
                    self.sequence_index = 0
                    self.token_buffer = []
                    self.text_buffer = ""

                    # Collect content based on current state
                    if self.current_state == "think":
                        reasoning_parts.append(buffered_content)
                    else:
                        content_parts.append(buffered_content)
                else:
                    # No buffered content, send normally
                    if self.current_state == "think":
                        reasoning_parts.append(token_text)
                    else:
                        content_parts.append(token_text)

        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)
        if not reasoning and not content:
            # If no content to send in this delta
            return None
        return DeltaMessage(reasoning=reasoning or None, content=content or None)
