# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for the Kimi K3 (XTML) chat format.

This strips the ``think`` channel out of generated text and hands the remainder
(``response`` + ``tools`` channels) downstream. Kimi K3 wraps the thinking
channel as an XTML element built from special tokens::

    <|open|>think<|sep|> <reasoning> <|close|>think<|sep|>

Two subtleties drive the implementation:
  * Unlike Kimi-K2 (a single ``<think>`` token), each K3 marker is a 3-token
    sequence, so the token-id helpers search for the marker *subsequence*
    rather than a single id.
  * In thinking mode the serving layer may feed ``<|open|>think<|sep|>`` as the
    generation prefix, so the model's output can begin *inside* the think
    channel with no open marker. The text paths therefore treat a missing open
    marker as "reasoning starts at offset 0".

When thinking is disabled (``chat_template_kwargs={"thinking": False}`` or
``{"enable_thinking": False}``, i.e. instruct mode) the parser returns every
delta as normal content; there is simply no think channel to extract.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _subseq_index(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Return start index of the last occurrence of needle in haystack, or -1."""
    n = len(needle)
    if n == 0:
        return -1
    for i in range(len(haystack) - n, -1, -1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return -1


class KimiK3ReasoningParser(ReasoningParser):
    """Reasoning parser for the Kimi K3 (XTML) think channel."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        # thinking can be disabled via chat_template_kwargs -> identity fallthrough
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        if thinking is None:
            thinking = chat_kwargs.get("enable_thinking", True)
        self._thinking_enabled = bool(thinking)

        # XTML markers as literal strings (skip_special_tokens=False at serve time)
        self._think_open = "<|open|>think<|sep|>"
        self._think_close = "<|close|>think<|sep|>"

        # Content-channel markers that must be stripped from the final output.
        # The tool parser handles these when active, but when tool_choice is
        # "none" (the default for requests without tools) the tool parser is
        # bypassed and the reasoning parser must strip them itself.
        self._response_open = "<|open|>response<|sep|>"
        self._response_close = "<|close|>response<|sep|>"
        self._message_close = "<|close|>message<|sep|>"

        # Tolerant matchers: defense-in-depth against vLLM's added-token spacing
        # ("<|open|> think <|sep|>"). The `\s*` is a no-op on clean input, so the
        # normal path stays byte-exact; it only helps if some serving path leaves
        # spaces_between_special_tokens on. See adjust_request for the real fix.
        open_marker = r"<\|open\|>"
        close_marker = r"<\|close\|>"
        sep_marker = r"<\|sep\|>"
        self._think_open_re = re.compile(open_marker + r"\s*think\s*" + sep_marker)
        self._think_close_re = re.compile(close_marker + r"\s*think\s*" + sep_marker)
        self._response_open_re = re.compile(
            open_marker + r"\s*response\s*" + sep_marker
        )
        self._response_close_re = re.compile(
            close_marker + r"\s*response\s*" + sep_marker
        )
        self._message_close_re = re.compile(
            close_marker + r"\s*message\s*" + sep_marker
        )

        # marker token-id subsequences (each marker is 3 tokens)
        self._think_open_ids = tokenizer.encode(
            self._think_open, add_special_tokens=False
        )
        self._think_close_ids = tokenizer.encode(
            self._think_close, add_special_tokens=False
        )
        self._last_streaming_delta_token_ids: tuple[int, ...] | None = None
        self._last_streaming_content_token_ids: list[int] | None = None

    @property
    def reasoning_start_str(self) -> str | None:
        return self._think_open

    @property
    def reasoning_end_str(self) -> str | None:
        return self._think_close

    def adjust_request(
        self,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self._thinking_enabled:
            return True
        # Reasoning has ended only if the *most recent* think block is closed:
        # the last close marker must come after the last open marker. A plain
        # "a close marker exists anywhere" check false-positives in multi-turn /
        # agent continuations, where the chat template keeps a prior turn's
        # think channel (with its <|close|>think<|sep|>) in the prompt while the
        # current turn is still reasoning (its <|open|>think<|sep|> is the newest
        # marker). See _subseq_index (returns the last occurrence).
        last_close = _subseq_index(input_ids, self._think_close_ids)
        last_open = _subseq_index(input_ids, self._think_open_ids)
        if last_open == -1:
            # No open marker in scope (e.g. the open was consumed as the
            # generation prefix): a close marker means reasoning has ended.
            return last_close != -1
        return last_close > last_open

    def _extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._thinking_enabled:
            return input_ids
        idx = _subseq_index(input_ids, self._think_close_ids)
        if idx == -1:
            return []  # still reasoning
        return input_ids[idx + len(self._think_close_ids) :]

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        cached_delta_ids = self._last_streaming_delta_token_ids
        cached_content_ids = self._last_streaming_content_token_ids
        self._last_streaming_delta_token_ids = None
        self._last_streaming_content_token_ids = None
        if cached_delta_ids == tuple(input_ids) and cached_content_ids is not None:
            return cached_content_ids
        return self._extract_content_ids(input_ids)

    def _strip_content_wrapper(self, text: str) -> str:
        """Strip ``<|open|>response<|sep|>…<|close|>response<|sep|>`` wrapper and
        ``<|close|>message<|sep|>`` from *text*.

        When Kimi K3 tool parsing is active it needs the raw XTML
        ``response`` + ``tools`` channels. Otherwise the reasoning parser cleans
        up the response wrapper itself so API users do not see XTML markers.
        """
        # Try structured unwrap first: extract body of response channel
        m_ro = self._response_open_re.search(text)
        m_rc = self._response_close_re.search(text, m_ro.end() if m_ro else 0)
        if m_ro is not None and m_rc is not None:
            text = text[m_ro.end() : m_rc.start()]
        elif m_ro is not None:
            # response opened but not closed (truncated)
            text = text[m_ro.end() :]
        else:
            # No response wrapper — strip stray markers if present
            text = self._response_open_re.sub("", text)
            text = self._response_close_re.sub("", text)
        text = self._message_close_re.sub("", text)
        return text

    @staticmethod
    def _should_preserve_tool_channels(
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> bool:
        return bool(getattr(request, "tools", None)) and (
            getattr(request, "tool_choice", None) != "none"
        )

    def _content_after_reasoning(
        self,
        text: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> str | None:
        if self._should_preserve_tool_channels(request):
            return text or None
        return self._strip_content_wrapper(text) or None

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """Split full text into ``(reasoning, rest)`` for the non-streaming path.

        Handles three shapes:
          * no think channel at all   -> ``(None, model_output)`` (all content)
          * open marker present       -> reasoning starts after ``<|open|>think<|sep|>``
          * open marker absent but a   close marker exists (gen-prefix consumed
            the open) -> reasoning starts at offset 0
        ``rest`` is whatever follows the close marker, fed on to the tool parser.
        """
        if not self._thinking_enabled:
            return None, self._content_after_reasoning(model_output, request)

        m_open = self._think_open_re.search(model_output)
        # reasoning content begins right after think-open (or at start if the
        # open marker was already consumed as a generation prefix)
        content_start = m_open.end() if m_open is not None else 0
        # if there is no think channel at all, everything is content
        if m_open is None and self._think_close_re.search(model_output) is None:
            return None, self._content_after_reasoning(model_output, request)

        m_close = self._think_close_re.search(model_output, content_start)
        if m_close is not None:
            reasoning = model_output[content_start : m_close.start()]
            rest = model_output[m_close.end() :]
            return (reasoning or None, self._content_after_reasoning(rest, request))
        # think not closed -> still reasoning, no content yet
        return (model_output[content_start:] or None, None)

    def _reasoning_text_ready_to_emit(self, text: str) -> str:
        """Return the reasoning prefix that is safe to stream now.

        Work from accumulated text, not from the current delta alone. That turns
        split-open and split-close handling into the same prefix-diff problem.

        Example 1: split think-open marker.
          chunks: ``<|open|>`` / ``think`` / ``<|sep|>reasoning``
          current text after chunk 1: ``<|open|>`` -> emit ``""``
          current text after chunk 2: ``<|open|>think`` -> emit ``""``
          current text after chunk 3: ``<|open|>think<|sep|>reasoning``
          -> emit ``reasoning``

        Example 2: split think-close marker.
          chunks: ``reasoning`` / ``<|close|>`` / ``think<|sep|>...``
          after ``<|close|>``, the suffix is only a partial close marker, so the
          sendable reasoning is still ``reasoning`` and the delta is empty. Once
          ``think<|sep|>`` arrives, the close branch hands the following response
          or tools text to the downstream parser.
        """
        m_open = self._think_open_re.search(text)
        if m_open is not None:
            text = text[m_open.end() :]
        overlap = 0
        for marker in (self._think_open, self._think_close):
            max_check = min(len(marker) - 1, len(text))
            for n in range(max_check, 0, -1):
                if text.endswith(marker[:n]):
                    overlap = max(overlap, n)
                    break
        return text[:-overlap] if overlap else text

    def _content_ready_to_emit(self, text: str) -> str:
        """Return the content prefix that is safe to stream now.

        Mirrors ``_reasoning_text_ready_to_emit`` but for the post-reasoning
        content phase.  Strips the ``<|open|>response<|sep|>`` prefix, holds back
        any partial marker suffix, and removes complete
        ``<|close|>response<|sep|>`` / ``<|close|>message<|sep|>`` markers.
        """
        # Strip response-open prefix
        m_open = self._response_open_re.search(text)
        if m_open is not None:
            text = text[m_open.end() :]

        # Remove complete close/message markers
        text = self._response_close_re.sub("", text)
        text = self._message_close_re.sub("", text)

        # Hold back partial markers at the end
        overlap = 0
        for marker in (
            self._response_open,
            self._response_close,
            self._message_close,
        ):
            max_check = min(len(marker) - 1, len(text))
            for n in range(max_check, 0, -1):
                if text.endswith(marker[:n]):
                    overlap = max(overlap, n)
                    break
        return text[:-overlap] if overlap else text

    def strip_content_streaming(
        self,
        previous_text: str,
        current_text: str,
    ) -> DeltaMessage | None:
        """Strip XTML content wrappers from streaming deltas after reasoning.

        Called by KimiK3Parser when no tool parser is configured, so the
        reasoning parser handles ``<|open|>response<|sep|>`` /
        ``<|close|>response<|sep|>`` / ``<|close|>message<|sep|>`` stripping itself.

        Works from accumulated text (``previous_text`` / ``current_text``
        already contain only post-reasoning content).
        """
        current_safe = self._content_ready_to_emit(current_text)
        previous_safe = self._content_ready_to_emit(previous_text)
        if current_safe.startswith(previous_safe):
            delta = current_safe[len(previous_safe) :]
        else:
            delta = current_safe
        return DeltaMessage(content=delta) if delta else None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        self._last_streaming_delta_token_ids = None
        self._last_streaming_content_token_ids = None
        if not self._thinking_enabled:
            return DeltaMessage(content=delta_text)

        # reasoning already ended -> downstream content
        if self._think_close_re.search(previous_text):
            return DeltaMessage(content=delta_text)

        # the close marker completes within this delta's accumulated text:
        # split the buffer at the close marker into reasoning vs trailing content.
        m_close = self._think_close_re.search(current_text)
        if m_close is not None:
            self._last_streaming_delta_token_ids = tuple(delta_token_ids)
            self._last_streaming_content_token_ids = self._extract_content_ids(
                list(current_token_ids)
            )
            m_open = self._think_open_re.search(current_text)
            r_start = m_open.end() if m_open is not None else 0
            reasoning = current_text[r_start : m_close.start()]
            already_sent = self._reasoning_text_ready_to_emit(previous_text)
            if reasoning.startswith(already_sent):
                reasoning_delta = reasoning[len(already_sent) :]
            else:
                reasoning_delta = reasoning
            content = current_text[m_close.end() :]
            return DeltaMessage(
                reasoning=reasoning_delta or None,
                content=content or None,
            )

        current_reasoning = self._reasoning_text_ready_to_emit(current_text)
        previous_reasoning = self._reasoning_text_ready_to_emit(previous_text)
        if current_reasoning.startswith(previous_reasoning):
            reasoning_delta = current_reasoning[len(previous_reasoning) :]
        else:
            reasoning_delta = current_reasoning
        if not reasoning_delta:
            return None
        return DeltaMessage(reasoning=reasoning_delta)

    # Backward-compatible aliases for existing unit tests and downstream users
    # that still call the pre-split method names.
    extract_reasoning_content = extract_reasoning
    extract_reasoning_content_streaming = extract_reasoning_streaming
