# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Optional, Union

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("granite")
class GraniteReasoningParser(ReasoningParser):
    """
    Reasoning parser for IBM Granite.

    IBM granite models currently use "Here is my thought process:"
    and "Here is my response:" to separate its thinking / response outputs.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        # NOTE: There have been some observed occurrences of quantized
        # instances of the current models using "Here's" instead of "Here is",
        # so to be safe, we match on both.
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

        # Substrings to match for sequence boundaries on raw text
        self.seq_boundary_end = ":"
        self.seq_boundary_start = "Here"

        # The longest any thinking / start of response message can be
        self.longest_think_start = max(
            len(think_start) for think_start in self.valid_think_starts)

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

        This increases the complexity of correctly handling streams, since we
        need to watch for specific sequences and correctly parse them without
        dropping content that is potentially overlapping & spanning multiple
        delta messages.

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
        reasoning_content, resp_seq_len, content = self._get_content_sections(
            current_text)
        # Either we haven't finished the start of the reasoning sequence,
        # or the model is generating something unexpected.
        if not reasoning_content:
            delta_message = self._get_delta_message_with_no_reasoning_bounds(
                current_text, delta_text)
        # We have a start of reasoning message, but have not yet finished
        # the start of response sequence.
        elif not content:
            delta_message = self._get_delta_message_with_no_response_bounds(
                current_text, reasoning_content, delta_text)
        # We've finished both the start of reasoning and start of response seq.
        else:
            # This should never happen since we matched on the response
            assert resp_seq_len is not None
            delta_message = self._get_delta_message_with_both_bounds(
                delta_text, reasoning_content, content, current_text,
                resp_seq_len)
        if not delta_message.content and not delta_message.reasoning_content:
            return None
        return delta_message

    #### Implementation details of stream parsing for granite models
    def _is_reasoning_start_substr(self, text: str) -> bool:
        """Check if a text matches one of the possible start reasoning seqs.

        Args:
            text (str): Text to check for leading substr.
        
        Returns:
            bool: True if any of the possible reasoning start seqs match.
        """
        return any(
            think_start.startswith(text)
            for think_start in self.valid_think_starts)

    def _is_response_start_substr(self, text: str) -> bool:
        """Check if a text matches one of the possible start response seqs.

        Args:
            text (str): Text to check for leading substr.
        
        Returns:
            bool: True if any of the possible response start seqs match.
        """
        return any(
            response_start.startswith(text)
            for response_start in self.valid_response_starts)

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
        prev_longest_length = len(current_text) - len(delta_text)
        is_substr = self._is_reasoning_start_substr(current_text)
        was_substr = self._is_reasoning_start_substr(
            current_text[:prev_longest_length])

        # Check if we just generated something NOT in the special token seq;
        # if so, add everything that we previously skipped with this delta
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
        self,
        current_text: str,
        reasoning_content: str,
        delta_text: str,
    ) -> DeltaMessage:
        """Parse the delta message when the current text has both reasoning
        content with no (response) content. NOTE that we may have overlapping
        tokens with the start of reasoning / start of response sequences on
        either side of the delta text.

        Args:
            current_text (str): The full previous + delta text.
            reasoning_content (str): reasoning content from current_text.
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

        # Check the state of potential start of response substring matches.
        prev_was_substr = self._is_response_start_substr(
            previous_text[prev_idx:]) if prev_idx >= 0 else False
        delta_continues_substr = self._is_response_start_substr(
            current_text[prev_idx:]) if prev_idx >= 0 else False
        delta_new_substr = self._is_response_start_substr(
            delta_text[delta_idx:]) if delta_idx >= 0 else False

        # Delta only contains potential continued response sequence text.
        if delta_continues_substr:
            return DeltaMessage(reasoning_content=None, content=None)

        if not prev_was_substr:
            # Delta may be starting a new response seq but has other text too.
            if delta_new_substr:
                return DeltaMessage(reasoning_content=delta_text[:delta_idx],
                                    content=None)
            # Normal case for most reasoning text (no potential special seqs).
            return DeltaMessage(reasoning_content=delta_text, content=None)
        # The substring that previously seemed to be a potential response
        # seq wasn't one; we need to add the content to the delta message,
        # and also slice off the potential response sequence
        elif delta_new_substr:
            reasoning_content = previous_text[
                prev_idx:] + delta_text[:delta_idx]
            return DeltaMessage(reasoning_content=reasoning_content,
                                content=None)
        # No new substring yet, and we broke our old one; take the whole delta
        return DeltaMessage(
            reasoning_content=previous_text[prev_idx:] + delta_text,
            content=None,
        )

    def _get_delta_message_with_both_bounds(
        self,
        delta_text: str,
        reasoning_content: str,
        response_content: str,
        current_text: str,
        response_seq_len: int,
    ) -> DeltaMessage:
        """Parse the delta message when the current text has both reasoning
        content and normal (response) content.

        Args:
            delta_text (str): Text to consider and parse content from.
            reasoning_content (str): reasoning content from current_text.
            response_content (str): response content from current_text.
            current_text (str): The full previous + delta text.
            response_seq_len(str): Len of the complete response sequence used.

        Returns:
            DeltaMessage: Message containing the parsed content.
        """
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
        self, current_text: str
    ) -> tuple[Optional[str], Optional[int], Optional[str]]:
        """Parse the text to extract the reasoning content / content
        if we have them.

        Args:
            current_text (str): The full previous + delta text.

        Returns:
            tuple[Optional[str], Optional[int], Optional[str]]: Tuple of len 3
            containing the reasoning content, the length of the response seq
            (if there is one) and the non-reasoning content.
        """
        current_chunk_start = 0
        start_reasoning_content = None
        parsed_content = False
        delimiter_idxs = [
            idx for idx, char in enumerate(current_text)
            if char == self.seq_boundary_end
        ]

        for current_chunk_end in delimiter_idxs:
            current_chunk = current_text[current_chunk_start:current_chunk_end]
            # Check to see if the start of reasoning seq if complete
            if start_reasoning_content is None:
                for think_start in self.valid_think_starts:
                    if current_chunk == think_start[:-1]:
                        start_reasoning_content = current_chunk_end + 1
                        current_chunk_start = current_chunk_end + 1
                        break

            # Check to see if the start of response seq if complete
            elif not parsed_content:
                for response_start in self.valid_response_starts:
                    if current_chunk[-len(response_start) +
                                     1:] == response_start[:-1]:
                        # Mark end of reasoning and start response content
                        # after the start of response sequence.
                        end_reasoning_content = current_chunk_end - len(
                            response_start)
                        reasoning_content = current_text[
                            start_reasoning_content:end_reasoning_content]
                        response_content = current_text[current_chunk_end + 1:]
                        return reasoning_content, len(
                            response_start), response_content

        if start_reasoning_content and not parsed_content:
            return current_text[start_reasoning_content:], None, None
        return None, None, None
