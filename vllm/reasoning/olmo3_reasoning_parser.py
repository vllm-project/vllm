# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


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
        super().__init__(tokenizer, *args, **kwargs)
        self.think_start_expr = r"<think>"
        self.think_end_expr = r"</think>"

        self.reasoning_regex = re.compile(
            rf"(?:{self.think_start_expr}\n*(.*?){self.think_end_expr})\n*(.*?)$",
            re.DOTALL,
        )

        self.think_start_ids = [14023, 771]
        self.think_end_ids = [524, 27963]
        self.bracket_with_newline_ids = {29, 397, 1363, 10586, 28801, 76819}

        # when state change, send out all the buffered text in last state
        self.buffered_text = []
        self.buffered_ids = []

        # self.current_state = "reasoning"
        # self.all_states = ["reasoning", "response"]

        # self.current_state = "idle"
        # self.expected_sequence = self.think_start_ids
        # # this sequence only for the think start, it has two way to start.
        # self.expected_sequence_side = self.think_end_ids_fast
        self.sequence_index = 0
        self.token_buffer = []
        self.text_buffer = ""

    # def is_reasoning_end(self, input_ids: list[int]) -> bool:
    #     return self.current_state == "response"

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # for Olmo 3 streaming reason parsing, the stream parse
        # will call first, and the same token will be called in
        # is_reasoning_end and extract_content_ids
        # this id is not part of content, so just return [] here.
        return []

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
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

        re_match = self.reasoning_regex.findall(model_output)
        if re_match:
            reasoning_content, response_content = re_match[0]
            if len(reasoning_content) == 0:
                reasoning_content = None
            if len(response_content) == 0:
                response_content = None
            return reasoning_content, response_content

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
        # Define sequences
        think_start_sequence = self.think_start_ids
        think_end_sequence = self.think_end_ids
        bracket_with_newline_sequence = self.bracket_with_newline_ids

        assert len(delta_token_ids) == 1
        # Process each token in the delta
        token = delta_token_ids[0]

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
                    and token
                    == self.expected_sequence_side[self.sequence_index - 1]
                ):
                    return self.sequence_index == len(
                        self.expected_sequence_side
                    )
                else:
                    return self.sequence_index == len(self.expected_sequence)
            else:
                return self.sequence_index == len(self.expected_sequence)

        # Check if token matches expected sequence
        token_in_state_seq = check_token_with_sequence(token)

        if token_in_state_seq:
            # Store matching token
            self.token_buffer.append(token)
            self.text_buffer += delta_text
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
                buffered_content = self.text_buffer + delta_text
                # Reset matching state
                self.sequence_index = 0
                self.token_buffer = []
                self.text_buffer = ""

                # Return content based on current state
                if self.current_state == "think":
                    return DeltaMessage(
                        reasoning_content=buffered_content, content=None
                    )
                else:
                    return DeltaMessage(
                        reasoning_content=None, content=buffered_content
                    )
            else:
                # No buffered content, send normally
                if self.current_state == "think":
                    return DeltaMessage(
                        reasoning_content=delta_text, content=None
                    )
                else:
                    return DeltaMessage(
                        reasoning_content=None, content=delta_text
                    )

        # If no content to send in this delta
        return None
