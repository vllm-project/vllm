# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Optional

import tokenizers
from packaging import version
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


class IncrementalDetokenizer:

    def __init__(self):
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids

    def update(self, new_token_ids: list[int],
               stop_terminated: bool) -> Optional[str]:
        self.token_ids.extend(new_token_ids)
        return None

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return ""

    @classmethod
    def from_new_request(
        cls,
        tokenizer: Optional[AnyTokenizer],
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        if (isinstance(tokenizer, PreTrainedTokenizerFast) and version.parse(
                tokenizers.__version__) >= version.parse("0.21.1")):
            # Fast tokenizer => use tokenizers library DecodeStream.
            # And only tokenizers >= 0.21.1 supports Fast Detokenizer.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):

    def __init__(self, request: EngineCoreRequest):
        super().__init__()

        # Stop strings
        params = request.sampling_params
        self.stop = stop = params.stop
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in stop) - 1
        else:
            self.stop_buffer_length = 0
        self._last_output_text_offset: int = 0

        # Generation data
        self.output_text = ""

    def update(self, new_token_ids: list[int],
               stop_terminated: bool) -> Optional[str]:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None

        # 1) Detokenize the new token ids incrementally.
        # TODO(woosuk): This method becomes very inefficient when the number of
        # new_token_ids is more than 1. We need to optimize this.
        offset_before = len(self.output_text)
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)

        if stop_terminated:
            if skipped_stop_token_id is not None:
                # Cleanup after skipping detokenization.
                self.token_ids.append(skipped_stop_token_id)
            # Stop token triggered; skip stop string check.
            return None

        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop:
            stop = StopChecker.check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - offset_before,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        raise NotImplementedError

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


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):

    def __init__(self, tokenizer: PreTrainedTokenizerFast,
                 request: EngineCoreRequest):
        super().__init__(request)

        sampling_params = request.sampling_params
        self.stream = DecodeStream(
            skip_special_tokens=sampling_params.skip_special_tokens)

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # Find a safe place to start.
        prompt_suffix = request.prompt_token_ids
        prompt_len = len(prompt_suffix)
        if prompt_len > 4:
            for i in range(4, min(prompt_len + 1, 24)):
                suffix = request.prompt_token_ids[-i:]
                if '�' not in self.tokenizer.decode(suffix):
                    prompt_suffix = suffix
                    break

        # Prime the stream.
        for tid in prompt_suffix:
            self.stream.step(self.tokenizer, tid)

        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens)

        if not self.spaces_between_special_tokens:
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            if (added_token_ids := getattr(self.tokenizer, "added_token_ids",
                                           None)) is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in
                    self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                # No added tokens.
                self.spaces_between_special_tokens = True

    def decode_next(self, next_token_id: int) -> str:
        token = self.stream.step(self.tokenizer, next_token_id)

        if not self.spaces_between_special_tokens:
            special_token = self.added_token_ids.get(next_token_id)
            is_special = special_token is not None
            if is_special and self.last_special:
                # Return raw token string without any prefixed spaces.
                token = special_token
            self.last_special = is_special

        return token or ""


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):

    def __init__(self, tokenizer: AnyTokenizer, request: EngineCoreRequest):
        super().__init__(request)

        self.tokenizer = tokenizer

        # Metadata for incremental detokenization.
        self.tokens, self.prefix_offset, self.read_offset = (
            convert_prompt_ids_to_tokens(
                tokenizer=tokenizer,
                prompt_ids=request.prompt_token_ids,
                skip_special_tokens=request.sampling_params.
                skip_special_tokens,
            ))

        self.token_ids.extend(request.prompt_token_ids)
        self.prompt_len = len(request.prompt_token_ids)

        params = request.sampling_params
        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = (
            params.spaces_between_special_tokens)

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids if not self.prompt_len else (
            self.token_ids[self.prompt_len:])

    def decode_next(self, next_token_id: int) -> str:
        new_tokens, decoded_text, prefix_offset, read_offset = (
            detokenize_incrementally(
                tokenizer=self.tokenizer,
                all_input_ids=self.token_ids,
                prev_tokens=self.tokens,
                prefix_offset=self.prefix_offset,
                read_offset=self.read_offset,
                skip_special_tokens=self.skip_special_tokens,
                spaces_between_special_tokens=self.
                spaces_between_special_tokens,
            ))

        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

        return decoded_text
