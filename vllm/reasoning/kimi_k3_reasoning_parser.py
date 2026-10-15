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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _match_at(haystack: Sequence[int], i: int, needle: Sequence[int]) -> bool:
    """Whether *needle* occurs in *haystack* starting at *i*.

    Compares element by element rather than slicing: ``haystack`` is usually a
    ``ConstantList``, whose slices cost a Python ``__getitem__`` plus a list
    allocation on every probe.
    """
    return all(haystack[i + k] == needle[k] for k in range(len(needle)))


def _subseq_index(
    haystack: Sequence[int], needle: Sequence[int], start: int = 0
) -> int:
    """Return the first occurrence of needle at or after start, or -1."""
    n = len(needle)
    if n == 0:
        return -1
    first = needle[0]
    for i in range(start, len(haystack) - n + 1):
        if haystack[i] == first and _match_at(haystack, i, needle):
            return i
    return -1


def _newest_marker(haystack: Sequence[int], a: Sequence[int], b: Sequence[int]) -> int:
    """Report which of *a* / *b* occurs last in *haystack*.

    Returns 0 if *a* is the newest marker, 1 if *b* is, -1 if neither occurs.
    A single backward pass stops at the newest marker instead of scanning
    the whole sequence separately for each marker.
    """
    if not a or not b:
        return -1
    a0, b0 = a[0], b[0]
    for i in range(len(haystack) - min(len(a), len(b)), -1, -1):
        head = haystack[i]
        if head == a0 and i + len(a) <= len(haystack) and _match_at(haystack, i, a):
            return 0
        if head == b0 and i + len(b) <= len(haystack) and _match_at(haystack, i, b):
            return 1
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
        self._response_open_ids = tokenizer.encode(
            self._response_open, add_special_tokens=False
        )
        self._streaming_reasoning_bounds: tuple[int, int] | None = None
        self._streaming_boundary_text: str | None = None
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
        # marker). A missing open marker (e.g. it was consumed as the generation
        # prefix) means a close marker alone ends reasoning, which is what
        # "close is the newest marker" already encodes.
        return (
            _newest_marker(input_ids, self._think_close_ids, self._think_open_ids) == 0
        )

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Iterable[int],
        *,
        prompt_len: int = 0,
    ) -> bool:
        """Reasoning-end check for a single decode step.

        The engine calls this once per structured-output request per decode step
        while the request is still inside the think channel, so it only has to
        look at the tokens generated this step, plus ``len(marker) - 1`` tokens
        of context in case a marker straddles the step boundary.

        The inherited default re-runs the full-sequence ``is_reasoning_end``,
        which makes the scheduler O(context) per request per step and starves the
        GPU on long agentic contexts.
        """
        if not self._thinking_enabled:
            return True
        delta = list(delta_ids)
        if not delta:
            return False
        carry = (
            max(
                len(self._think_close_ids),
                len(self._think_open_ids),
                len(self._response_open_ids),
            )
            - 1
        )
        head = len(input_ids) - len(delta)
        window_start = max(0, head - carry)
        window = list(input_ids[window_start:head]) + delta
        response_start = self.first_response_start(
            window, max(0, prompt_len - window_start)
        )
        if (
            response_start != -1
            and window_start + response_start + len(self._response_open_ids) > head
        ):
            return True
        return _newest_marker(window, self._think_close_ids, self._think_open_ids) == 0

    def _extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._thinking_enabled:
            return input_ids
        bounds = self._reasoning_bounds(input_ids)
        return input_ids[bounds[1] :] if bounds is not None else []

    def is_streaming_transition_ready(self, token_ids: list[int]) -> bool:
        """The frontend may hand off only after the delimiter text is visible."""
        if not self._thinking_enabled:
            return True
        return self._last_streaming_content_token_ids is not None and (
            self.is_reasoning_end_streaming(token_ids, token_ids)
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        cached_delta_ids = self._last_streaming_delta_token_ids
        cached_content_ids = self._last_streaming_content_token_ids
        self._last_streaming_delta_token_ids = None
        self._last_streaming_content_token_ids = None
        if cached_delta_ids == tuple(input_ids) and cached_content_ids is not None:
            return cached_content_ids
        return self._extract_content_ids(input_ids)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self._thinking_enabled:
            return 0
        response_start = self.first_response_start(token_ids)
        if response_start != -1:
            token_ids = token_ids[:response_start]
        think_open = self._think_open_ids
        think_close = self._think_close_ids
        first_ids = {m[0] for m in (think_open, think_close) if m}
        n = len(token_ids)
        count = 0
        in_reasoning = True
        i = 0
        while i < n:
            head = token_ids[i]
            if head in first_ids:
                if i + len(think_open) <= n and _match_at(token_ids, i, think_open):
                    in_reasoning = True
                    i += len(think_open)
                    continue
                if i + len(think_close) <= n and _match_at(token_ids, i, think_close):
                    in_reasoning = False
                    i += len(think_close)
                    continue
            if in_reasoning:
                count += 1
            i += 1
        return count

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

    def first_response_start(self, token_ids: Sequence[int], start: int = 0) -> int:
        """Find the first genuine response opener at or after the output boundary."""
        return _subseq_index(token_ids, self._response_open_ids, start)

    def _reasoning_bounds(self, token_ids: Sequence[int]) -> tuple[int, int] | None:
        response_start = self.first_response_start(token_ids)
        prefix = token_ids[:response_start] if response_start != -1 else token_ids
        close = _subseq_index(prefix, self._think_close_ids)
        if close != -1:
            return close, close + len(self._think_close_ids)
        if response_start != -1:
            return response_start, response_start
        return None

    def _split_reasoning(
        self,
        text: str,
        token_ids: Sequence[int],
        reasoning_end: int,
        content_start: int,
    ) -> tuple[str | None, str] | None:
        """Locate boundaries with IDs, retaining the engine's truncated text."""
        reasoning_start = 0
        if len(token_ids) >= len(self._think_open_ids) and _match_at(
            token_ids, 0, self._think_open_ids
        ):
            reasoning_start = len(self._think_open_ids)
        offsets = []
        for token_offset in (reasoning_start, reasoning_end, content_start):
            prefix = self.model_tokenizer.decode(
                token_ids[:token_offset], skip_special_tokens=False
            )
            visible = min(len(prefix), len(text))
            if text[:visible] != prefix[:visible]:
                return None
            offsets.append(visible)
        start, end, content = offsets
        return text[start:end] or None, text[content:]

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None]:
        """Split at a genuine channel boundary, retaining the engine's text.

        Without token IDs, preserve the existing text-based extraction.
        """
        if not self._thinking_enabled:
            return None, self._content_after_reasoning(model_output, request)
        if model_output_token_ids:
            bounds = self._reasoning_bounds(model_output_token_ids)
            if bounds is None:
                bounds = (len(model_output_token_ids), len(model_output_token_ids))
            parsed = self._split_reasoning(
                model_output, model_output_token_ids, *bounds
            )
            if parsed is not None:
                reasoning, content = parsed
                return reasoning, self._content_after_reasoning(content, request)

        m_open = self._think_open_re.match(model_output)
        # reasoning content begins right after think-open (or at start if the
        # open marker was already consumed as a generation prefix)
        content_start = m_open.end() if m_open is not None else 0

        m_close = self._think_close_re.search(model_output, content_start)
        if (
            m_open is None
            and m_close is None
            and not model_output_token_ids
            and self._response_open_re.search(model_output) is not None
        ):
            return None, self._content_after_reasoning(model_output, request)
        if m_close is not None:
            reasoning = model_output[content_start : m_close.start()]
            rest = model_output[m_close.end() :]
            return (reasoning or None, self._content_after_reasoning(rest, request))
        # think not closed -> still reasoning, no content yet
        return (model_output[content_start:] or None, None)

    def _reasoning_text_ready_to_emit(
        self, text: str, token_ids: Sequence[int] = ()
    ) -> str:
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
        m_open = self._think_open_re.match(text)
        if m_open is not None and (
            not token_ids
            or len(token_ids) >= len(self._think_open_ids)
            and _match_at(token_ids, 0, self._think_open_ids)
        ):
            text = text[m_open.end() :]
        overlap = 0
        markers = [self._think_open, self._think_close]
        if any(
            list(token_ids[-n:]) == self._response_open_ids[:n]
            for n in range(1, min(len(self._response_open_ids), len(token_ids) + 1))
        ):
            markers.append(self._response_open)
        for marker in markers:
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

        bounds = self._streaming_reasoning_bounds
        first_boundary = bounds is None
        if first_boundary:
            carry = max(len(self._think_close_ids), len(self._response_open_ids)) - 1
            window_start = max(0, len(current_token_ids) - len(delta_token_ids) - carry)
            local = self._reasoning_bounds(current_token_ids[window_start:])
            if local is not None:
                bounds = (window_start + local[0], window_start + local[1])
                self._streaming_reasoning_bounds = bounds
                transition_end = bounds[1]
                if bounds[0] == bounds[1]:
                    transition_end += len(self._response_open_ids)
                self._streaming_boundary_text = self.model_tokenizer.decode(
                    current_token_ids[:transition_end], skip_special_tokens=False
                )
        if bounds is not None:
            boundary_text = self._streaming_boundary_text
            assert boundary_text is not None
            if previous_text.startswith(boundary_text):
                return DeltaMessage(content=delta_text)
            parsed = self._split_reasoning(current_text, current_token_ids, *bounds)
            previous = self._split_reasoning(previous_text, current_token_ids, *bounds)
            if parsed is not None and previous is not None:
                reasoning, content = parsed
                already_sent = (
                    self._reasoning_text_ready_to_emit(
                        previous_text, previous_token_ids
                    )
                    if first_boundary
                    else previous[0] or ""
                )
                reasoning_delta = (reasoning or "")[len(already_sent) :]
                # Stop-string buffering can expose IDs before their visible text.
                if current_text.startswith(boundary_text):
                    self._last_streaming_delta_token_ids = tuple(delta_token_ids)
                    self._last_streaming_content_token_ids = list(
                        current_token_ids[bounds[1] :]
                    )
                else:
                    content = ""
                return DeltaMessage(
                    reasoning=reasoning_delta or None, content=content or None
                )

        # reasoning already ended -> downstream content
        if not current_token_ids and self._think_close_re.search(previous_text):
            return DeltaMessage(content=delta_text)

        # the close marker completes within this delta's accumulated text:
        # split the buffer at the close marker into reasoning vs trailing content.
        m_close = (
            self._think_close_re.search(current_text) if not current_token_ids else None
        )
        if m_close is not None:
            self._last_streaming_delta_token_ids = tuple(delta_token_ids)
            self._last_streaming_content_token_ids = []
            m_open = self._think_open_re.match(current_text)
            r_start = m_open.end() if m_open is not None else 0
            reasoning = current_text[r_start : m_close.start()]
            already_sent = self._reasoning_text_ready_to_emit(
                previous_text, previous_token_ids
            )
            if reasoning.startswith(already_sent):
                reasoning_delta = reasoning[len(already_sent) :]
            else:
                reasoning_delta = reasoning
            content = current_text[m_close.end() :]
            return DeltaMessage(
                reasoning=reasoning_delta or None,
                content=content or None,
            )

        current_reasoning = self._reasoning_text_ready_to_emit(
            current_text, current_token_ids
        )
        previous_reasoning = self._reasoning_text_ready_to_emit(
            previous_text, previous_token_ids
        )
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
