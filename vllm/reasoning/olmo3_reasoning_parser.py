# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
import dataclasses as dt
from typing import Optional, Union, List, Tuple
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

        # if self.text_buffer.endswith("thoughts"):
        #     breakpoint()

        # if start_think_idx >= 0 and end_think_idx >= 0:
        #     # special case here to handle when both <think>
        #     # and </think> are in the buffer. in that case,
        #     # we extract whatever it is between the two
        #     # tokens, and then reset the buffer to end of
        #     # end thinking token.
        #     pretext = self.text_buffer[
        #         start_think_idx + len(self.think_start) : end_think_idx
        #     ]
        #     self.state = Olmo3ReasoningState.CONTENT
        #     breakpoint()
        #     self.text_buffer = self.text_buffer[
        #         end_think_idx + len(self.think_end) :
        #     ]
        #     # the None is necessary in case there is no text
        #     # between the <think> and </think> tokens
        #     return DeltaMessage(reasoning_content=pretext or None)

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

    #     # We use this buffer text in case we need to skip one or more delta texts
    #     # that could be part of special tokens.
    #     # self.buffer_text: str = ""

    # def find_think_start(self, text: str) -> int:
    #     idx = text.find(self.think_start_expr)
    #     if idx >= 0:
    #         return idx + len(self.think_start_expr)
    #     return idx

    # def find_think_end(self, text: str) -> int:
    #     return text.rfind(self.think_end_expr)

    # def delta_might_be_think_start(self, text: str) -> bool:
    #     if len(text) >= len(self.think_start_expr):
    #         return self.think_start_expr in text
    #     else:
    #         for idx in range(0, len(self.think_start_expr), len(text)):
    #             if text == self.think_start_expr[idx : idx + len(text)]:
    #                 return True
    #     return False

    # def delta_might_be_think_end(self, text: str) -> bool:
    #     if len(text) >= len(self.think_end_expr):
    #         return self.think_end_expr in text
    #     else:
    #         for idx in range(0, len(self.think_end_expr), len(text)):
    #             if text == self.think_end_expr[idx : idx + len(text)]:
    #                 return True
    #     return False

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

        # maybe_delta = self.buffer.process(delta_text)
        # if maybe_delta is None and len(self.buffer) > 0:
        #     return self.buffer.process(None)

        # # we add buffer to delta_text
        # if len(self.buffer_text) > 0:
        #     delta_text = self.buffer_text + delta_text
        #     self.buffer_text = ""

        # if (
        #     delta_text in self.think_start_expr
        #     or delta_text in self.think_end_expr
        #     or delta_text in self.no_think_expr
        # ):
        #     # there's a chance that this content is actually part of reasoning
        #     # tokens, so for now we save it in a buffer
        #     self.buffer_text += delta_text

        #     return None

        # end_think_idx = current_text.rfind(self.think_end_expr)
        # start_think_idx = current_text.find(self.think_start_expr)
        # delta_start_idx = len(current_text) - len(delta_text)

        # if start_think_idx >= 0 and delta_start_idx < start_think_idx:
        #     # the message is starting before the beginning of thinking, but a
        #     # thinking token has already been found! in that case, we
        #     # emit everything before the thinking token, and save the rest
        #     # in the buffer
        #     content = current_text[:delta_start_idx]
        #     self.buffer_text += current_text[delta_start_idx:]
        #     return DeltaMessage(content=content or None)
        # elif end_think_idx >= 0 and delta_start_idx < end_think_idx:
        #     # you found an end token, and the delta message starts
        #     # before the end token
        #     think_start_delta_idx = max(
        #         start_think_idx + len(self.think_start_expr), delta_start_idx
        #     )
        #     think_end_delta_idx = min(len(current_text), end_think_idx)
        #     reasoning_content = current_text[
        #         think_start_delta_idx:think_end_delta_idx
        #     ]
        #     self.buffer_text += current_text[think_end_delta_idx:]
        #     return DeltaMessage(reasoning_content=reasoning_content or None)
        # elif end_think_idx >= 0:
        #     delta_think_idx = max(
        #         delta_start_idx, end_think_idx + len(self.think_end_expr)
        #     )
        #     # the delta text is fully after the end token,
        #     # so we can emit it as-is
        #     return DeltaMessage(content=current_text[delta_think_idx:] or None)
        # elif start_think_idx >= 0:
        #     # the delta text is fully after the start of thinking token,
        #     # emit as a reasoning message
        #     delta_think_idx = max(
        #         delta_start_idx, start_think_idx + len(self.think_start_expr)
        #     )
        #     return DeltaMessage(
        #         reasoning_content=current_text[delta_think_idx:] or None
        #     )

        # return None

        # # if end_think_idx >= 0 and delta_start_idx < end_think_idx:
        # #     # check if there is any text in delta_text before end_think_idx;
        # #     # if yes, just return that; save any of the remaining text in buffer
        # #     reasoning_content = current_text[delta_start_idx:end_think_idx]
        # #     self.buffer_text += current_text[
        # #         end_think_idx + len(self.think_end_expr) :
        # #     ]
        # #     return DeltaMessage(reasoning_content=reasoning_content)

        # # elif end_think_idx >= 0:
        # #     start_content = max(
        # #         end_think_idx + len(self.think_end_expr), delta_start_idx
        # #     )
        # #     content = current_text[start_content:]
        # #     return DeltaMessage(content=content)

        # # elif start_think_idx >= 0 and delta_start_idx < start_think_idx:
        # #     # check if there is a content before the start of thinking
        # #     # if there is, return it as a DeltaMessage, and save the
        # #     # remaining text for later processing
        # #     content = current_text[delta_start_idx:start_think_idx]
        # #     self.buffer_text += current_text[
        # #         start_think_idx + len(self.think_start_expr) :
        # #     ]
        # #     return DeltaMessage(content=content)

        # # elif start_think_idx >= 0:
        # #     start_content = max(
        # #         start_think_idx + len(self.think_start_expr),
        # #         delta_start_idx,
        # #     )
        # #     reason_content = current_text[start_content:]
        # #     return DeltaMessage(reasoning_content=reason_content)

        # # return None

        # # # if start_think_token_idx >= 0:
        # # #     start_content = max(
        # # #         start_think_token_idx + len(self.think_start_expr),
        # # #         delta_start_idx,
        # # #     )
        # # #     content = current_text[start_content:]
        # # #     return DeltaMessage(content=content)

        # # # if start_think_token_idx >= 0 and e
        # # #     # before thinking
        # # #     if delta_message_start_idx < delta_message_start_idx:
        # # #         # we emit only the part of the delta that is before the start of thinking
        # # #         text = current_text[
        # # #             delta_message_start_idx:start_think_token_idx
        # # #         ]
        # # #         return DeltaMessage(content=text)
        # # #     else:
        # # #         # fully outside of thinking
        # # #         return DeltaMessage(content=delta_text)

        # # # if end_think_token_idx > 0:
        # # #     # after thinking
        # # #     if delta_message_start_idx < delta_message_start_idx:
        # # #         # we emit only the part of the delta that is before the end of thinking
        # # #         text = current_text[delta_message_start_idx:end_think_token_idx]
        # # #         return DeltaMessage(thinking_content=text)
        # # #     else:
        # # #         # fully outside of thinking
        # # #         return DeltaMessage(content=delta_text)

        # # # # if self.find_think_end(current_text) > 0:
        # # # #     delta_message = DeltaMessage(content=delta_text)

        # # # # if self.find_think_start(current_text) > 0:
        # # # #     delta_message = DeltaMessage(reasoning_content=delta_text)

        # # # # print(delta_message)
        # # # # print("====")
        # # # # return delta_message

        # # # # print("current_text:", current_text)
        # # # # print("delta_text:", delta_text)

        # # # # # this is where the new text starts
        # # # # delta_start_idx = len(current_text) - len(delta_text)

        # # # # # find boundary of where the model has started to think.
        # # # # start_think = self.find_think_start(current_text)
        # # # # end_think = self.find_think_end(current_text)

        # # # # print("start_think:", start_think)
        # # # # print("end_think:", end_think)
        # # # # print("delta_start_idx:", delta_start_idx)

        # # # # delta_message: Union[DeltaMessage, None] = None

        # # # # if start_think >= 0 and end_think >= 0:
        # # # #     # the model is done thinking. the question is: did it stop
        # # # #     # in this delta, or was it already done?
        # # # #     # to answer this, we check if end_think is smaller than
        # # # #     # delta_start_idx (was already done) or not (just finished)

        # # # #     if end_think < delta_start_idx:
        # # # #         # the model was already done thinking; delta_text is all content
        # # # #         delta_message = DeltaMessage(content=delta_text)
        # # # #     else:
        # # # #         # the model has just finished thinking; delta_text is a mix
        # # # #         # of reasoning and content
        # # # #         reasoning_content = current_text[delta_start_idx:end_think]
        # # # #         content = current_text[end_think + len(self.think_end_expr) :]
        # # # #         delta_message = DeltaMessage(
        # # # #             reasoning_content=reasoning_content or None,
        # # # #             content=content or None,
        # # # #         )

        # # # # elif start_think >= 0:
        # # # #     # the model is thinking! same as before... did it just start?
        # # # #     if start_think < delta_start_idx:
        # # # #         # the model was already thinking; delta_text is all content
        # # # #         delta_message = DeltaMessage(reasoning_content=delta_text)
        # # # #     else:
        # # # #         # the model has just started thinking; delta_text is a mix
        # # # #         # of thinking and content

        # # # #         reasoning_content = current_text[
        # # # #             delta_start_idx : start_think - len(self.think_start_expr)
        # # # #         ]
        # # # #         content = current_text[start_think:]
        # # # #         delta_message = DeltaMessage(
        # # # #             reasoning_content=reasoning_content or None,
        # # # #             content=content or None,
        # # # #         )

        # # # # print(delta_message)
        # # # # print("====")
        # # # # return delta_message
