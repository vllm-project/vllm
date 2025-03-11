# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.v1.engine import EngineCoreRequest, FinishReason

logger = init_logger(__name__)


@dataclass
class IncrementalDetokenizer:

    # Generation data
    token_ids: list[int]
    output_text: str = ""
    tokens: list[str] = field(default_factory=list)
    prompt_len: int = 0

    # Stop strings
    stop: list[str] = field(default_factory=list)
    include_stop_str_in_output: bool = False

    # Metadata for incremental detokenization
    prefix_offset: int = 0
    read_offset: int = 0

    # Parameters for detokenization
    eos_token_id: Optional[int] = None
    stop_token_ids: Optional[list[int]] = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True

    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: Optional[AnyTokenizer] = None

    # Accounting for stop string buffering
    stop_buffer_length: int = 0
    _last_output_text_offset: int = 0

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids if not self.prompt_len else (
            self.token_ids[self.prompt_len:])

    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":

        if tokenizer is None:
            return cls(token_ids=[])

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
            stop_token_ids=request.sampling_params.stop_token_ids,
            ignore_eos=request.sampling_params.ignore_eos,
            eos_token_id=request.eos_token_id,
        )

    def _check_stop_string(
        self,
        step_decoded_text,
    ) -> Optional[str]:
        """Check for stop-string; truncate if triggered.
        
        Args:
          step_decoded_text: all decoded text in this step

        Returns:
          stop string if triggered, otherwise `None`
        """
        # 3) Evaluate stop-string criteria.
        stop_string = None
        if self.stop:
            stop = StopChecker.check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(step_decoded_text),
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    def _detokenize(self) -> str:
        """One pass of incremental detokenization."""
        (new_tokens, new_decoded_token_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             tokenizer=self.tokenizer,
             all_input_ids=self.token_ids,
             prev_tokens=self.tokens,
             prefix_offset=self.prefix_offset,
             read_offset=self.read_offset,
             skip_special_tokens=self.skip_special_tokens,
             spaces_between_special_tokens=self.spaces_between_special_tokens,
         )
        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset
        return new_decoded_token_text

    def _assert_valid_stop_token(self, stop_token_id: int) -> None:
        """Token must be EOS (if not ignoring) or stop token"""
        if ((stop_token_id != self.eos_token_id or self.ignore_eos)
                and stop_token_id not in (self.stop_token_ids or ())):
            raise AssertionError(f"Engine core finish reason is STOP but "
                                 f"{stop_token_id} is not in stop_token_ids "
                                 f"= {self.stop_token_ids}")

    def update(self, new_token_ids: list[int],
               core_finish_reason: Optional[FinishReason]) -> Optional[str]:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """
        # Skip detokenization
        if self.tokenizer is None:
            self.token_ids.extend(new_token_ids)
            return None
        if not new_token_ids:
            return None

        # 1) Detokenize the new token ids incrementally.
        # TODO(woosuk): This method becomes very inefficient when the number of
        # new_token_ids is more than 1. We need to optimize this.
        decoded_text = ""
        for new_token_id in new_token_ids[0:-1]:
            self.token_ids.append(new_token_id)
            decoded_text += self._detokenize()

        # 2) Deferred text truncation for engine core EOS/stop-token checks.
        #    Skip detokenization of last token if it is a stop-token.
        new_token_id = new_token_ids[-1]
        self.token_ids.append(new_token_id)
        if is_stop_token := core_finish_reason == FinishReason.STOP:
            self._assert_valid_stop_token(new_token_id)
        if not is_stop_token or self.include_stop_str_in_output:
            decoded_text += self._detokenize()

        self.output_text += decoded_text

        if is_stop_token:
            # EOS/stop-token triggered
            return None

        # 3) Evaluate stop-string criteria.
        return self._check_stop_string(decoded_text)

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
