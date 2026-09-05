# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-content parser for MuseGlimmer channel-scoped output."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping, Sequence
from functools import cached_property
from weakref import WeakKeyDictionary

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.muse_glimmer_utils import (
    advance_emitted,
    current_assistant_turn,
    open_recipient,
    safe_open_body,
    visible_channels,
)

# Every channel header ends with this marker, so `is_reasoning_end` can only
# flip False->True on a decode step that completes it. That makes it a sound
# anchor for the O(len(delta)) prefilter in `is_reasoning_end_streaming`.
_CHANNEL_MARKER = "<|message|>"
# Tokens decoded to confirm that a non-atomic candidate really completed
# _CHANNEL_MARKER. Only has to span the marker plus the token carrying its
# first character, so anything above ~4 is slack.
_CONFIRM_TAIL_TOKENS = 16
# The completer-id scan costs one pass over the vocabulary, and the
# structured-output manager builds a parser per request, so the result is
# shared per tokenizer. Weak keys let the entry go when the tokenizer does.
_MARKER_COMPLETER_CACHE: MutableMapping[object, frozenset[int]] = WeakKeyDictionary()


class MuseGlimmerReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self._emitted_reasoning = ""
        self._emitted_content = ""
        self._initial_recipient: str | None = None

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Preserve MuseGlimmer framing so channel parsing remains possible."""
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Report an open answer or tool channel to engine-side callers."""
        try:
            text = self.model_tokenizer.decode(input_ids)
        except Exception:
            return False
        recipient = open_recipient(current_assistant_turn(text))
        return recipient not in (None, "self")

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """Answer `is_reasoning_end` without decoding on ordinary steps.

        The structured-output manager asks this once per decode step per
        running request until reasoning ends, and `is_reasoning_end` decodes
        the whole sequence, which makes decoding quadratic in sequence length.
        Generated text is append-only and a channel cannot open without
        completing `_CHANNEL_MARKER`, so a step whose delta cannot supply that
        marker's final character cannot be the step the answer flips on and is
        rejected in O(len(delta_ids)) without decoding.

        The result is not monotonic: after a channel closes, a later step may
        answer False again. Callers latch the first True
        (`StructuredOutputManager.should_advance` sets `reasoning_ended`), and
        the guarantee kept here is that no False->True transition of
        `is_reasoning_end` is ever skipped.
        """
        completers = self._channel_marker_completers
        if completers:
            num_delta = 0
            saw_marker = False
            saw_completer = False
            for token_id in delta_ids:
                num_delta += 1
                if token_id == self._channel_marker_id:
                    saw_marker = True
                elif token_id in completers:
                    saw_completer = True
            # An empty delta carries no new text to judge, so fall through to
            # the full check rather than assuming False.
            if num_delta and not saw_marker:
                if not saw_completer:
                    return False
                if not self._tail_shows_marker(input_ids, num_delta):
                    return False
        return self.is_reasoning_end(input_ids)

    @cached_property
    def _channel_marker_id(self) -> int | None:
        """`_CHANNEL_MARKER`'s own id, when the tokenizer has it as one token."""
        try:
            return self.vocab.get(_CHANNEL_MARKER)
        except Exception:
            return None

    @cached_property
    def _channel_marker_completers(self) -> frozenset[int]:
        tokenizer = self.model_tokenizer
        try:
            cached = _MARKER_COMPLETER_CACHE.get(tokenizer)
            if cached is None:
                cached = self._marker_completer_ids(_CHANNEL_MARKER)
                _MARKER_COMPLETER_CACHE[tokenizer] = cached
            return cached
        except TypeError:
            # Tokenizer not weak-referenceable; correctness does not depend on
            # the cache.
            return self._marker_completer_ids(_CHANNEL_MARKER)

    def _marker_completer_ids(self, marker: str) -> frozenset[int]:
        """Token ids that could supply `marker`'s final character.

        A token can complete the marker only if its text starts with a
        non-empty suffix of the marker (finishing a marker begun in earlier
        tokens) or contains the marker outright. This deliberately does not
        assume the marker arrives as its single special token: 2,730 distinct
        token-id sequences decode to ``to=self<|message|>`` in the MuseGlimmer
        tokenizer, and nothing constrains generation to the canonical
        spelling. Suffix-overlap collection covers every spelling without
        enumerating any.

        Returns an empty set if the vocabulary is unavailable, which disables
        the prefilter rather than risking a missed transition.
        """
        try:
            vocab = self.vocab
        except Exception:
            return frozenset()
        suffixes = tuple(marker[i:] for i in range(len(marker)))
        return frozenset(
            token_id
            for text, token_id in vocab.items()
            if text and (text.startswith(suffixes) or marker in text)
        )

    def _tail_shows_marker(self, input_ids: Sequence[int], num_delta: int) -> bool:
        """Whether `_CHANNEL_MARKER` is present near the end of the sequence.

        Only ever used to reject a step whose delta held a completer id but no
        marker. Decode failures return True so the step falls through to the
        full check, keeping the no-skipped-transition guarantee.
        """
        window = _CONFIRM_TAIL_TOKENS + num_delta
        tail_ids = input_ids[-window:] if len(input_ids) > window else input_ids
        try:
            return _CHANNEL_MARKER in self.model_tokenizer.decode(tail_ids)
        except Exception:
            return True

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        """Continue classifying generation in the prompt's open channel."""
        try:
            text = self.model_tokenizer.decode(prompt_token_ids)
        except Exception:
            return
        self._initial_recipient = open_recipient(current_assistant_turn(text))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Content-id slicing is unreliable for multi-token markers.
        return []

    def _seeded_text(self, text: str) -> str:
        if self._initial_recipient is None:
            return text
        return f"to={self._initial_recipient}<|message|>{text}"

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        """Promote any unstreamed answer body when generation is truncated."""
        content, _reasoning, _content_open, _reasoning_open = visible_channels(
            self._seeded_text(previous_text)
        )
        remainder = content[len(self._emitted_content) :]
        return remainder or None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning while preserving framed text for channel consumers."""
        _content, reasoning, _content_open, _reasoning_open = visible_channels(
            model_output
        )
        return reasoning or None, model_output or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Stream clean reasoning and answer bodies from the shared segmenter."""
        content, reasoning, content_open, reasoning_open = visible_channels(
            self._seeded_text(current_text)
        )
        if content_open:
            content = safe_open_body(content)
        if reasoning_open:
            reasoning = safe_open_body(reasoning)

        reasoning_delta, self._emitted_reasoning = advance_emitted(
            self._emitted_reasoning, reasoning
        )
        content_delta, self._emitted_content = advance_emitted(
            self._emitted_content, content
        )
        if not reasoning_delta and not content_delta:
            return None

        return DeltaMessage(
            reasoning=reasoning_delta or None,
            content=content_delta or None,
        )
