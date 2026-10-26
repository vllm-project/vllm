# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from collections.abc import Callable

import tokenizers.decoders
from tokenizers import Tokenizer
from transformers import TokenizersBackend

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"

ContentFilter = Callable[[str, list[int], bool], str | None]


def uses_fast_detokenizer(tokenizer: TokenizerLike) -> bool:
    """Whether the engine detokenizes with `FastIncrementalDetokenizer`."""
    return isinstance(tokenizer, TokenizersBackend)


class IncrementalDetokenizer:
    def __init__(self):
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids)

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        self.token_ids.extend(new_token_ids)
        return None

    def get_next_output(self, finished: bool, delta: bool) -> tuple[str, int]:
        """Return the output text plus the number of trailing output tokens whose
        text it holds back."""
        return "", 0

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        content_filter: ContentFilter | None = None,
    ) -> "IncrementalDetokenizer":
        assert request.sampling_params is not None

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        if uses_fast_detokenizer(tokenizer):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request, content_filter)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request, content_filter)


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    def __init__(
        self,
        request: EngineCoreRequest,
        content_filter: ContentFilter | None = None,
    ):
        super().__init__()

        # Stop strings
        params = request.sampling_params
        assert params is not None
        if params.stop is None:
            self.stop = []
        elif isinstance(params.stop, str):
            self.stop = [params.stop]
        else:
            self.stop = params.stop
        self.min_tokens = params.min_tokens
        self.include_stop_str_in_output = params.include_stop_str_in_output

        self._content_filter = content_filter if self.stop else None
        self._content_text = ""
        self._content_starts: list[int] = []
        self._content_output_starts: list[int] = []
        self._content_output_end = 0

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if self.stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.stop_buffer_length = 0
        self._last_output_text_offset: int = 0
        self._held_token_ends: list[int] | None = (
            [] if self.stop_buffer_length else None
        )
        self._released_text_end = 0

        # Generation data
        self.output_text = ""

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """Update RequestState for the request_id by:
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
        num_prev_tokens = self.num_output_tokens()
        first_new_token_index = len(self.token_ids)
        # Output text offsets at the boundaries of the new tokens.
        token_offsets = [len(self.output_text)] if self.stop else None
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
            if token_offsets is not None:
                token_offsets.append(len(self.output_text))

        if skipped_stop_token_id is not None:
            # Cleanup after skipping detokenization.
            self.token_ids.append(skipped_stop_token_id)
            if token_offsets is not None:
                token_offsets.append(len(self.output_text))

        if (held_ends := self._held_token_ends) is not None:
            assert token_offsets is not None
            held_ends.extend(token_offsets[1:])

        # 2) Evaluate stop strings.
        if token_offsets is None:
            return None
        # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
        stop_check_offset = None
        if self.num_output_tokens() > self.min_tokens:
            stop_check_offset = token_offsets[max(self.min_tokens - num_prev_tokens, 0)]
        stop = None
        if self._content_filter is not None:
            stop = self._find_content_stop(
                token_offsets[0], new_token_ids, stop_terminated, stop_check_offset
            )
        if self._content_filter is None and stop_check_offset is not None:
            stop = self._find_stop(stop_check_offset)
        if stop is None:
            return None

        stop_string, truncate_to, stop_end = stop
        # Drop tokens generated after the one completing the stop string
        # (possible when multiple tokens are appended in one step).
        num_kept = (
            bisect_left(token_offsets, stop_end, lo=1)
            if stop_end > token_offsets[0]
            else 0
        )
        del self.token_ids[first_new_token_index + num_kept :]
        if held_ends is not None:
            del held_ends[len(held_ends) - len(token_offsets) + 1 + num_kept :]
        if truncate_to != -1:
            self.output_text = self.output_text[:truncate_to]
            if held_ends is not None:
                # Clamp to the truncated text, keeping the ends sorted
                # if a streaming-input session continues after the stop.
                held_ends[:] = [min(end, truncate_to) for end in held_ends]
        return stop_string

    def _find_stop(self, stop_check_offset: int) -> tuple[str, int, int] | None:
        stop = check_stop_strings(
            output_text=self.output_text,
            new_char_count=len(self.output_text) - stop_check_offset,
            stop=self.stop,
            include_in_output=self.include_stop_str_in_output,
        )
        if stop is None:
            return None
        stop_string, truncate_to = stop
        stop_end = len(self.output_text) if truncate_to == -1 else truncate_to
        if not self.include_stop_str_in_output:
            stop_end += len(stop_string)
        return stop_string, truncate_to, stop_end

    def _find_content_stop(
        self,
        delta_start: int,
        new_token_ids: list[int],
        finished: bool,
        stop_check_offset: int | None,
    ) -> tuple[str, int, int] | None:
        assert self._content_filter is not None
        try:
            content = self._content_filter(
                self.output_text[delta_start:], new_token_ids, finished
            )
        except Exception:
            logger.exception("Content filter failed; matching all output text.")
            self._content_filter = None
            return None
        if not content:
            return None
        output_start = self._add_content(content)
        if stop_check_offset is None:
            return None

        # In the step that reaches min_tokens, skip content generated before
        # it. Content the filter releases late was generated past it.
        new_char_count = len(content)
        if stop_check_offset > delta_start:
            new_char_count -= max(stop_check_offset - output_start, 0)
        stop = check_stop_strings(
            output_text=self._content_text,
            new_char_count=max(new_char_count, 0),
            stop=self.stop,
            include_in_output=self.include_stop_str_in_output,
        )
        if stop is None:
            return None
        stop_string, truncate_to = stop
        if self.include_stop_str_in_output:
            if truncate_to == -1:
                truncate_to = len(self._content_text)
            stop_end = truncate_to = self._output_offset(truncate_to, is_end=True)
        else:
            stop_end = self._output_offset(truncate_to + len(stop_string), is_end=True)
            truncate_to = self._output_offset(truncate_to, is_end=False)
        if truncate_to >= len(self.output_text):
            truncate_to = -1
        return stop_string, truncate_to, stop_end

    def _add_content(self, content: str) -> int:
        text = self.output_text
        lowest = self._content_output_end
        # (offset in `content`, output text offset) of each contiguous run
        runs: list[tuple[int, int]] = []
        output_start = text.rfind(content, lowest)
        if output_start != -1:
            runs.append((0, output_start))
        else:
            positions: list[int] = []
            i = len(text) - 1
            for char in reversed(content):
                while i >= lowest and text[i] != char:
                    i -= 1
                if i < lowest:
                    break
                positions.append(i)
                i -= 1
            if positions:
                positions.reverse()
                # Leading content missing from the output text, e.g. added by
                # the parser, maps to where the rest starts.
                skipped = len(content) - len(positions)
                if skipped:
                    runs.append((0, positions[0]))
                runs += [
                    (skipped + offset, position)
                    for offset, position in enumerate(positions)
                    if offset == 0 or position != positions[offset - 1] + 1
                ]
            else:
                runs.append((0, max(lowest, len(text) - len(content))))

        for offset, output_offset in runs:
            self._content_starts.append(len(self._content_text) + offset)
            self._content_output_starts.append(output_offset)
        last_offset, last_output_offset = runs[-1]
        self._content_output_end = last_output_offset + len(content) - last_offset
        self._content_text += content
        return runs[0][1]

    def _output_offset(self, content_offset: int, is_end: bool) -> int:
        starts = self._content_starts
        output_starts = self._content_output_starts
        index = (
            bisect_left(starts, content_offset)
            if is_end
            else bisect_right(starts, content_offset)
        ) - 1
        output_offset = output_starts[index] + content_offset - starts[index]
        return min(output_offset, len(self.output_text))

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        raise NotImplementedError

    def get_next_output(self, finished: bool, delta: bool) -> tuple[str, int]:
        """If delta is True, only new text since the last call to
        this method is returned"""
        length = len(self.output_text)
        num_held_tokens = 0
        if held_ends := self._held_token_ends:
            if finished:
                held_ends.clear()
                self._released_text_end = length
            else:
                num_released = bisect_right(held_ends, length - self.stop_buffer_length)
                if num_released:
                    self._released_text_end = held_ends[num_released - 1]
                    del held_ends[:num_released]
                num_held_tokens = len(held_ends)
                length = self._released_text_end

        if not delta:
            return self.output_text[:length], num_held_tokens

        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length], num_held_tokens
        return "", num_held_tokens


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(
        self,
        tokenizer: TokenizersBackend,
        request: EngineCoreRequest,
        content_filter: ContentFilter | None = None,
    ):
        super().__init__(request, content_filter)

        sampling_params = request.sampling_params
        assert sampling_params is not None

        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # Use native prefill to prime the decode stream with prompt tokens.
        # Look up DecodeStream on the module so backend patches (e.g. the
        # fastokens shim that replaces ``tokenizers.decoders.DecodeStream``)
        # are honored regardless of import order.
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )

        self.spaces_between_special_tokens = (
            sampling_params.skip_special_tokens
            or sampling_params.spaces_between_special_tokens
        )

        if not self.spaces_between_special_tokens:
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            added_token_ids = getattr(self.tokenizer, "added_token_ids", None)
            if added_token_ids is None:
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content
                    for tid, tok in self.tokenizer.get_added_tokens_decoder().items()
                }

            if added_token_ids:
                self.last_special = False
                self.added_token_ids = added_token_ids
            else:
                # No added tokens.
                self.spaces_between_special_tokens = True

    def decode_next(self, next_token_id: int) -> str:
        token = self._protected_step(next_token_id)

        if not self.spaces_between_special_tokens:
            special_token = self.added_token_ids.get(next_token_id)
            is_special = special_token is not None
            if is_special and self.last_special:
                # Return raw token string without any prefixed spaces.
                token = special_token
            self.last_special = is_special

        return token or ""

    def _protected_step(self, next_token_id: int) -> str | None:
        try:
            token = self.stream.step(self.tokenizer, next_token_id)
        except (OverflowError, TypeError):
            # Handle rare observed overflow, still to be diagnosed.
            # See https://github.com/vllm-project/vllm/issues/21951.
            logger.exception("Encountered invalid token id: %r", next_token_id)
            token = None
        except Exception as e:
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):
                raise e
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.",
                self.request_id,
            )
            self.stream = tokenizers.decoders.DecodeStream(
                skip_special_tokens=self.skip_special_tokens
            )
            token = self.stream.step(self.tokenizer, next_token_id)
        return token


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        request: EngineCoreRequest,
        content_filter: ContentFilter | None = None,
    ):
        super().__init__(request, content_filter)

        self.tokenizer = tokenizer
        params = request.sampling_params
        assert params is not None

        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )

        # Metadata for incremental detokenization.
        if request.prompt_token_ids is not None:
            self.tokens, self.prefix_offset, self.read_offset = (
                convert_prompt_ids_to_tokens(
                    tokenizer=tokenizer,
                    prompt_ids=request.prompt_token_ids,
                    skip_special_tokens=params.skip_special_tokens,
                )
            )
        else:
            # Prompt embedding requests cannot be detokenized, in general.
            self.tokens = [""] * self.prompt_len
            self.prefix_offset = 0
            self.read_offset = 0

        self.token_ids.extend(request.prompt_token_ids or [0] * self.prompt_len)

        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = params.spaces_between_special_tokens

    @property
    def output_token_ids(self) -> list[int]:
        if self.prompt_len:
            return self.token_ids[self.prompt_len :]
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    def decode_next(self, next_token_id: int) -> str:
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
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

        return decoded_text


def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence
    output text accordingly.

    Returns tuple (stop_string, offset) if matched or else None.

    Where stop_string is the matched stop string and offset is the
    length to which output_text should be truncated, or -1 for no
    truncation.

    When several stop strings match within the newly generated text (for
    example when speculative decoding appends multiple tokens in a single
    step), the stop string that completes earliest in the text is selected,
    so the result matches appending one token at a time. Ties are broken by
    stop-list order.
    """
    if not new_char_count or not stop:
        return None

    best_stop_str: str | None = None
    best_stop_index = 0
    best_end = sys.maxsize
    for stop_str in stop:
        stop_string_len = len(stop_str)
        # Avoid searching already-searched text.
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue

        # Prefer the stop string that completes earliest in the text.
        end = stop_index + stop_string_len
        if end < best_end:
            best_stop_str = stop_str
            best_stop_index = stop_index
            best_end = end

    if best_stop_str is None:
        return None

    if include_in_output:
        # Truncate to end of stop string.
        if best_end >= len(output_text):
            # No truncation required.
            return best_stop_str, -1
        return best_stop_str, best_end

    # Truncate the output text to the beginning of the stop string.
    return best_stop_str, best_stop_index
