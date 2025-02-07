# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


@dataclass
class IncrementalDetokenizer:

    # Generation data
    output_text: str
    tokens: List[str]
    token_ids: List[int]
    prompt_len: int

    # Stop strings
    stop: List[str]
    include_stop_str_in_output: bool

    # Metadata for incremental detokenization
    prefix_offset: int
    read_offset: int

    # Parameters for detokenization
    skip_special_tokens: bool
    spaces_between_special_tokens: bool

    # Tokenizer for this request
    tokenizer: AnyTokenizer

    # Accounting for stop string buffering
    stop_buffer_length: int
    _last_output_text_offset: int = 0

    @property
    def output_token_ids(self) -> List[int]:
        return self.token_ids[self.prompt_len:]

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":

        tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
            tokenizer=tokenizer,
            prompt_ids=request.prompt_token_ids,
            skip_special_tokens=request.sampling_params.skip_special_tokens,
        )

        stops = request.sampling_params.stop
        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if stops and not request.sampling_params.include_stop_str_in_output:
            stop_buffer_length = max(len(s) for s in stops) - 1
        else:
            stop_buffer_length = 0

        return cls(
            output_text="",
            tokens=tokens,
            # Detokenizer mutates this list, so need a unique copy.
            # NOTE(Nick): could we take ownership of it though?
            token_ids=request.prompt_token_ids.copy(),
            stop=stops,
            include_stop_str_in_output=request.sampling_params.
            include_stop_str_in_output,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=request.sampling_params.skip_special_tokens,
            spaces_between_special_tokens=request.sampling_params.
            spaces_between_special_tokens,
            prompt_len=len(request.prompt_token_ids),
            tokenizer=tokenizer,
            stop_buffer_length=stop_buffer_length,
        )

    def update(self, new_token_ids: List[int]) -> Optional[str]:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """

        # 1) Detokenize the new token ids incrementally.
        # TODO(woosuk): This method becomes very inefficient when the number of
        # new_token_ids is more than 1. We need to optimize this.
        decoded_text = ""
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            (new_tokens, new_decoded_token_text, prefix_offset,
             read_offset) = detokenize_incrementally(
                 tokenizer=self.tokenizer,
                 all_input_ids=self.token_ids,
                 prev_tokens=self.tokens,
                 prefix_offset=self.prefix_offset,
                 read_offset=self.read_offset,
                 skip_special_tokens=self.skip_special_tokens,
                 spaces_between_special_tokens=self.
                 spaces_between_special_tokens,
             )

            self.tokens.extend(new_tokens)
            self.prefix_offset = prefix_offset
            self.read_offset = read_offset

            decoded_text += new_decoded_token_text

        self.output_text += decoded_text

        # 2) Evaluate stop criteria.
        stop_string = None
        if self.stop:
            stop = StopChecker.check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(decoded_text),
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            return self.output_text[:-buffer_length] if buffer_length else (
                self.output_text)
        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""
