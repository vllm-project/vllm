# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional, Sequence, Tuple, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.entrypoints.openai.reasoning_parsers.abs_reasoning_parsers import (
    ReasoningParser, ReasoningParserManager)
from vllm.logger import init_logger

logger = init_logger(__name__)


@ReasoningParserManager.register_module("granite")
class GraniteReasoningParser(ReasoningParser):
    """
    Reasoning parser for IBM Granite.

    IBM granite models currently use "Here is my thought process:"
    and "Here is my response:" to separate its thinking / response outputs.
    NOTE: There have been some observed occurrences of quantized instances of
    the current models using "Here's" instead of "Here is", so to be safe, we
    match on both
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_expr = r"(?:Here's|Here is) my thought process:"
        self.response_start_expr = r"(?:Here's|Here is) my response:"

        self.reasoning_regex = re.compile(
            rf"{self.think_start_expr}(.*?){self.response_start_expr}(.*)",
            re.DOTALL)

        self.valid_think_starts = [
            "Here's my thought process:", "Here is my thought process:"
        ]
        self.valid_response_starts = [
            "Here's my response:", "Here is my response:"
        ]
        self.seq_boundary_end = ":"
        self.seq_boundary_start = "Here"

        # The longest any thinking / start of response message can be
        self.longest_think_start = max(
            len(think_start) for think_start in self.valid_think_starts)

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract the reasoning content & content sections, respectively.
        If the sequence doesn't match what we expect, i.e., the model generates
        something else, all content is considered non-reasoning content.

        Args:
            model_output (str): Output of the model to be parsed.
            request (ChatCompletionReqest): Request being processed.

        Returns:
            Tuple[Optional[str], Optional[str]]: Tuple pair containing the
            reasoning content and non-reasoning content.
        """
        re_match = self.reasoning_regex.findall(model_output)
        if not re_match:
            return None, model_output
        reasoning_content, response_content = re_match[0]
        if not response_content:
            return reasoning_content, None
        return reasoning_content, response_content

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """Extract the reasoning content / content emitted by granite models;
        If the sequence doesn't match what we expect, i.e., the model generates
        something else, all content is considered non-reasoning content.

        NOTE: Granite models do not use a special token to start their reasoning
        and response sections; instead they have token sequences, e.g.,

                Here is my thought process: Foo Here is my response: Bar

        This increases the complexity of correctly handling streams by a lot,
        since we need to watch for specific sequences and correctly handle
        parsing them without dropping content that is potentially overlapping
        & spanning across multiple delta messages.

        Args:
            previous_text (str): Previous text outside of this delta message.
            current_text (str): Previous text + delta text.
            delta_text (str): Text to consider and parse content from.
            previous_token_ids (Sequence[int]): Token IDs of previous_text.
            current_token_ids (Sequence[int]): Token IDs of current_text.
            delta_token_ids (Sequence[int]): Token IDs of delta_text.

        Returns:
            Union[DeltaMessage, None]
                DeltaMessage with either reasoning content or content, or None.
        """
        reasoning_content, response_content = self._get_content_sections(
            current_text)
        if not reasoning_content:
            delta_message = self._get_delta_message_with_no_reasoning_bounds(
                current_text, delta_text)
        # We have a start of reasoning message, but not the response message yet
        elif not response_content:
            delta_message = self._get_delta_message_with_no_response_bounds(
                current_text, reasoning_content, delta_text)
        else:
            delta_message = self._get_delta_message_with_both_bounds(
                delta_text, reasoning_content, response_content, current_text)
        if not delta_message.content and not delta_message.reasoning_content:
            return None
        return delta_message

    #### Implementation details of stream parsing for granite models
    def _get_delta_message_with_no_reasoning_bounds(
        self,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage:
        """Parse the delta message when the current text has not yet completed
        its start of reasoning sequence.

        Args:
            current_text (str): The full previous + delta text.
            delta_text (str): Text to consider and parse content from.

        Returns:
            DeltaMessage: Message containing the parsed content.
        """
        # Even before this, we already had a longer text str than any expected
        # message; The output is not parsable; assume we rectified this
        # previously and just add the delta text to the content.
        prev_longest_length = len(current_text) - len(delta_text)
        if prev_longest_length > self.longest_think_start:
            return DeltaMessage(reasoning_content=None, content=delta_text)

        is_substr = any(current_text in think_start
                        for think_start in self.valid_think_starts)
        was_substr = any(current_text[:prev_longest_length] in think_start
                         for think_start in self.valid_think_starts)
        # Check if we just generated something NOT in the special token seq;
        # if so, we add everything that we previously skipped with this delta
        # message and append everything to content in the future.
        if was_substr and not is_substr:
            return DeltaMessage(
                reasoning_content=None,
                content=current_text,
            )
        if is_substr:
            # Might still be in the special token sequence; return nothing
            return DeltaMessage(reasoning_content=None, content=None)
        # Otherwise the sequence has already been broken and we already
        # corrected; just return the delta text as normal content.
        return DeltaMessage(reasoning_content=None, content=delta_text)

    def _get_delta_message_with_no_response_bounds(
            self, current_text: str, reasoning_content: str,
            delta_text: str) -> DeltaMessage:
        """Parse the delta message when the current text has both reasoning
        content with no (response) content. NOTE that we may have overlapping
        tokens with the start of reasoning / start of response sequences on
        either side of the delta text.

        Args:
            current_text (str): The full previous + delta text.
            reasoning_content (str): reasoning content parsed from current_text.
            delta_text (str): Text to consider and parse content from.

        Returns:
            DeltaMessage: Message containing the parsed content.
        """
        # If we have no reasoning content or explicitly end with the start of
        # response sequence, we are in transition to the response; need to be
        # careful here, since the final token (:) will match the reasoning
        # content and fully parse it out; we should not pass the : back.
        ends_with_start_response_seq = any(
            current_text.endswith(response_start)
            for response_start in self.valid_response_starts)
        if reasoning_content is None or ends_with_start_response_seq:
            return DeltaMessage(reasoning_content=None, content=None)

        # Consider previous / current text only within context of the reasoning
        previous_text = reasoning_content[:-len(delta_text)]
        current_text = reasoning_content

        # We need to be careful about adding unfinished response sequences;
        # Find the place at which we MIGHT be starting a response sequence
        prev_idx = previous_text.rfind(self.seq_boundary_start)
        delta_idx = delta_text.rfind(self.seq_boundary_start)

        # And check if either we had one in the previous text, or have on
        prev_was_substr = any(
            previous_text[prev_idx:] in response_start for response_start in
            self.valid_response_starts) if prev_idx >= 0 else False
        delta_continues_substr = any(
            current_text[prev_idx:] in response_start for response_start in
            self.valid_response_starts) if prev_idx >= 0 else False
        delta_new_substr = any(
            delta_text[delta_idx:] in response_start for response_start in
            self.valid_response_starts) if delta_idx >= 0 else False
        # It was a substring before, and all delta tokens are
        # part of the potential start of response sequence

        if prev_was_substr and delta_continues_substr:
            return DeltaMessage(reasoning_content=None, content=None)

        if not prev_was_substr:
            # Don't add the potential unfinished response sequence
            if delta_new_substr:
                return DeltaMessage(reasoning_content=delta_text[:delta_idx],
                                    content=None)
            # No possible places to start the response seq; continue normally
            return DeltaMessage(reasoning_content=delta_text, content=None)
        # The substring being continued is the same one as before;
        # the whole delta message is part of the potential response seq
        if delta_continues_substr:
            return DeltaMessage(reasoning_content=None, content=None)
        # The substring that previously seemed to be a potential response
        # seq wasn't one; we need to add the content to the delta message,
        # and also slice off the potential response sequence
        elif delta_new_substr:
            return DeltaMessage(reasoning_content=previous_text[prev_idx:] +
                                delta_text[:delta_idx],
                                content=None)
        # No new substring yet, and we broke our old one; take the whole delta
        return DeltaMessage(reasoning_content=previous_text[prev_idx:] +
                            delta_text,
                            content=None)

    def _get_delta_message_with_both_bounds(
        self,
        delta_text: str,
        reasoning_content: str,
        response_content: str,
        current_text: str,
    ) -> DeltaMessage:
        """Parse the delta message when the current text has both reasoning
        content and normal (response) content.

        Args:
            delta_text (str): Text to consider and parse content from.
            reasoning_content (str): reasoning content parsed from current_text.
            response_content (str): response content parsed from current_text.
            current_text (str): The full previous + delta text.

        Returns:
            DeltaMessage: Message containing the parsed content.
        """
        # We have reasoning and response content, but it may not all be in the
        # delta text; we need to consider the length of the start of response
        # sequence and divide just the delta text part.
        ##### HACK pass this through
        for rs in self.valid_response_starts:
            if rs in current_text:
                response_seq_len = len(rs)
                break
        #####

        # Always have content; take length to the end
        delta_content = delta_text[-len(response_content):]
        reasoning_end_idx = len(delta_text) - (len(response_content) +
                                               response_seq_len)
        if reasoning_end_idx < 0:
            delta_reasoning_content = None
        else:
            # Get the starting offset
            start_reasoning_content_idx = len(
                reasoning_content) + response_seq_len + len(
                    response_content) - 1
            delta_offset = len(current_text) - len(delta_text)
            start_offset = start_reasoning_content_idx - delta_offset
            if start_offset < 0:
                start_offset = 0
            delta_reasoning_content = delta_text[
                start_offset:reasoning_end_idx]

        return DeltaMessage(
            reasoning_content=delta_reasoning_content,
            content=delta_content,
        )

    def _get_content_sections(
            self, current_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse the text to extract the reasoning content / content
        if we have them.

        Args:
            current_text (str): The full previous + delta text.

        Returns:
            Tuple[Optional[str], Optional[str]]: Tuple pair containing the
            reasoning content and non-reasoning content.
        """
        delimiter_idxs = [
            idx for idx, char in enumerate(current_text)
            if char == self.seq_boundary_end
        ]
        current_chunk_start = 0
        start_reasoning_content = None
        start_response_content = None

        for current_chunk_end in delimiter_idxs:
            current_chunk = current_text[current_chunk_start:current_chunk_end]
            # Check to see if this is start of reasoning
            if start_reasoning_content is None:
                for think_start in self.valid_think_starts:
                    if current_chunk == think_start[:-1]:
                        start_reasoning_content = current_chunk_end + 1
                        current_chunk_start = current_chunk_end + 1
                        break

            # Check to see if this is start of response
            elif start_response_content is None:
                for response_start in self.valid_response_starts:
                    if current_chunk[-len(response_start) +
                                     1:] == response_start[:-1]:
                        end_reasoning_content = current_chunk_end - len(
                            response_start)
                        reasoning_content = current_text[
                            start_reasoning_content:end_reasoning_content]
                        response_content = current_text[current_chunk_end + 1:]
                        # Ensure we handle empty strings / None consistently
                        if not response_content:
                            response_content = None
                        return reasoning_content, response_content
        # Set the actual response content
        if start_reasoning_content and start_response_content is None:
            return current_text[start_reasoning_content:], None
        return None, None
