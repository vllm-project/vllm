# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for the Kimi K3 (XTML) chat format.

This strips the `think` channel out of generated text and hands the remainder
(`response` + `tools` channels) downstream. Kimi K3 wraps the thinking
channel as an XTML element built from special tokens::

    <|open|>think<|sep|> <reasoning> <|close|>think<|sep|>

Two subtleties drive the implementation:
  * Unlike Kimi-K2 (a single `<think>` token), each K3 marker is a 3-token
    sequence, so the token-id helpers search for the marker *subsequence*
    rather than a single id.
  * In thinking mode the serving layer may feed `<|open|>think<|sep|>` as the
    generation prefix, so the model's output can begin *inside* the think
    channel with no open marker. The text paths therefore treat a missing open
    marker as "reasoning starts at offset 0".
  * The model does not always close the think channel the prompt opened. It
    may open the response directly (`<|open|>response<|sep|>`), or, mostly on
    ~1M-token prompts, write the reply straight away and end it with
    `<|close|>response<|sep|><|close|>message<|sep|>`. Either marker inside
    the think channel leaves it: text before a response opener is reasoning,
    text before a response closer is the reply and is returned as content
    (same reading as sgl-project/sglang#37743). These markers are located by
    token id, so reasoning that merely quotes one is left alone.

When thinking is disabled (`chat_template_kwargs={"thinking": False}` or
`{"enable_thinking": False}`, i.e. instruct mode) the parser returns every
delta as normal content; there is simply no think channel to extract.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import regex as re
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser
from vllm.v1.token_buffer import rfind_token_sequences

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.parser.engine.parser_engine import ReasoningEnd


def _match_at(haystack: Sequence[int], i: int, needle: Sequence[int]) -> bool:
    """Whether *needle* occurs in *haystack* starting at *i*.

    Compares element by element rather than slicing: `haystack` is usually a
    `ConstantList`, whose slices cost a Python `__getitem__` plus a list
    allocation on every probe.
    """
    return all(haystack[i + k] == needle[k] for k in range(len(needle)))


def _subseq_index(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Return start index of the last occurrence of needle in haystack, or -1."""
    n = len(needle)
    if n == 0:
        return -1
    first = needle[0]
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i] == first and _match_at(haystack, i, needle):
            return i
    return -1


def _newest_marker(haystack: Sequence[int], a: Sequence[int], b: Sequence[int]) -> int:
    """Report which of *a* / *b* occurs last in *haystack*.

    Returns 0 if *a* is the newest marker, 1 if *b* is, -1 if neither occurs.
    Equivalent to comparing two `_subseq_index` results, but a single backward
    pass that stops at the first hit instead of walking to index 0 twice.
    """
    return -1 if not a or not b else rfind_token_sequences(haystack, (a, b))[1]


# Kinds of marker that leave the think channel.
_THINK_CLOSE, _RESPONSE_OPEN, _RESPONSE_CLOSE = range(3)


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
        self._tools_open_re = re.compile(open_marker + r"\s*tools\s*" + sep_marker)

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
        self._response_close_ids = tokenizer.encode(
            self._response_close, add_special_tokens=False
        )
        # Markers that leave the think channel, indexed by _THINK_CLOSE etc.
        self._exit_ids = (
            self._think_close_ids,
            self._response_open_ids,
            self._response_close_ids,
        )
        self._exit_res = (
            self._think_close_re,
            self._response_open_re,
            self._response_close_re,
        )
        self._last_streaming_delta_token_ids: tuple[int, ...] | None = None
        self._last_streaming_content_token_ids: list[int] | None = None
        # Streaming: tokens already scanned for a think exit, the exit found
        # (token index, kind), and whether its text has been handed off.
        self._scanned = 0
        self._exit_index = -1
        self._exit_kind = -1
        self._exited = False

    def _first_think_exit(
        self, token_ids: Sequence[int], start: int = 0
    ) -> tuple[int, int]:
        """`(index, kind)` of the first marker leaving the think channel."""
        heads = {m[0] for m in self._exit_ids if m}
        n = len(token_ids)
        for i in range(start, n):
            if token_ids[i] in heads:
                for kind, marker in enumerate(self._exit_ids):
                    if (
                        marker
                        and i + len(marker) <= n
                        and _match_at(token_ids, i, marker)
                    ):
                        return i, kind
        return -1, -1

    def _exit_text_match(
        self, text: str, token_ids: Sequence[int], index: int, kind: int
    ) -> "re.Match[str] | None":
        """Locate in *text* the marker that starts at token *index*.

        Reasoning may quote the same marker as ordinary text before it; those
        spellings are counted in the decoded prefix and skipped.
        """
        marker_re = self._exit_res[kind]
        prefix = self.model_tokenizer.decode(
            list(token_ids[:index]), skip_special_tokens=False
        )
        quoted = len(marker_re.findall(prefix))
        for i, match in enumerate(marker_re.finditer(text)):
            if i == quoted:
                return match
        return None

    def _skipped_think_channel(self, text: str, pos: int = 0) -> bool:
        """A response-close inside the think channel before any think exit.

        Text-only fallback for callers that pass no token ids.
        """
        m_rc = self._response_close_re.search(text, pos)
        if m_rc is None:
            return False
        for exit_re in (
            self._think_close_re,
            self._response_open_re,
            self._tools_open_re,
        ):
            m = exit_re.search(text, pos)
            if m is not None and m.start() < m_rc.start():
                return False
        return True

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
        return self._reasoning_end_newest(input_ids)

    def _reasoning_end_newest(self, window: Sequence[int]) -> bool:
        """True when the newest marker leaves the current think channel.

        That is `<|close|>think<|sep|>`, or a response opener or closer when
        the model left the channel without closing it (see the module
        docstring). It only counts when it is newer than the last
        `<|open|>think<|sep|>`, so a prior turn's markers kept in the prompt do
        not end the current turn's reasoning.
        """
        open_ids = self._think_open_ids
        return any(_newest_marker(window, ids, open_ids) == 0 for ids in self._exit_ids)

    def find_reasoning_end(self, token_ids: Sequence[int]) -> "ReasoningEnd | None":
        from vllm.parser.engine.parser_engine import ReasoningEnd

        if not self._thinking_enabled:
            return ReasoningEnd(0, False)
        close = self._think_close_ids
        # A context-free scan is safe only when the streaming carry cannot
        # contain a complete close, nor finish one at the start of this delta.
        if not close or len(close) != len(self._think_open_ids):
            return None
        num_tokens = len(token_ids)
        for offset in range(1, len(close)):
            remaining = len(close) - offset
            if (
                remaining <= num_tokens
                and token_ids[0] == close[offset]
                and all(token_ids[i] == close[offset + i] for i in range(remaining))
            ):
                return None
        for i in range(num_tokens - len(close) + 1):
            if token_ids[i] == close[0] and _match_at(token_ids, i, close):
                return ReasoningEnd(i + len(close) - 1, False)
        return ReasoningEnd(num_tokens, False)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """Reasoning-end check for a single decode step.

        The engine calls this once per structured-output request per decode step
        while the request is still inside the think channel, so it only has to
        look at the tokens generated this step, plus `len(marker) - 1` tokens
        of context in case a marker straddles the step boundary.

        The inherited default re-runs the full-sequence `is_reasoning_end`,
        which makes the scheduler O(context) per request per step and starves the
        GPU on long agentic contexts.
        """
        if not self._thinking_enabled:
            return True
        delta = list(delta_ids)
        if not delta:
            return False
        carry = max(map(len, (self._think_open_ids, *self._exit_ids))) - 1
        head = len(input_ids) - len(delta)
        window = list(input_ids[max(0, head - carry) : head]) + delta
        return self._reasoning_end_newest(window)

    def _extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._thinking_enabled:
            return input_ids
        idx = _subseq_index(input_ids, self._think_close_ids)
        if idx != -1:
            return input_ids[idx + len(self._think_close_ids) :]
        idx, kind = self._first_think_exit(input_ids)
        if kind == _RESPONSE_OPEN:
            return input_ids[idx:]
        if kind == _RESPONSE_CLOSE:
            # Skipped think channel: the reply is everything after the last
            # think-open (or from the start when that was the generation prefix).
            last_open = _subseq_index(input_ids[:idx], self._think_open_ids)
            start = last_open + len(self._think_open_ids) if last_open != -1 else 0
            return input_ids[start:]
        return []  # still reasoning

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
        think_open = self._think_open_ids
        think_close = self._think_close_ids
        response_open = self._response_open_ids
        response_close = self._response_close_ids
        first_ids = {
            m[0] for m in (think_open, think_close, response_open, response_close) if m
        }
        n = len(token_ids)
        count = 0
        channel_count = 0  # tokens counted since the think channel opened
        in_reasoning = True
        i = 0
        while i < n:
            head = token_ids[i]
            if head in first_ids:
                if i + len(think_open) <= n and _match_at(token_ids, i, think_open):
                    in_reasoning = True
                    channel_count = 0
                    i += len(think_open)
                    continue
                if i + len(think_close) <= n and _match_at(token_ids, i, think_close):
                    in_reasoning = False
                    i += len(think_close)
                    continue
                if (
                    in_reasoning
                    and response_close
                    and i + len(response_close) <= n
                    and _match_at(token_ids, i, response_close)
                ):
                    # Skipped think channel: what it held was the reply.
                    count -= channel_count
                    in_reasoning = False
                    i += len(response_close)
                    continue
                if i + len(response_open) <= n and _match_at(
                    token_ids, i, response_open
                ):
                    in_reasoning = False
                    i += len(response_open)
                    continue
            if in_reasoning:
                count += 1
                channel_count += 1
            i += 1
        return count

    def _strip_content_wrapper(self, text: str) -> str:
        """Strip `<|open|>response<|sep|>…<|close|>response<|sep|>` wrapper and
        `<|close|>message<|sep|>` from *text*.

        When Kimi K3 tool parsing is active it needs the raw XTML
        `response` + `tools` channels. Otherwise the reasoning parser cleans
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
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
        token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None]:
        """Split full text into `(reasoning, rest)` for the non-streaming path.

        Handles these shapes:
          * response opener without think markers -> no think channel; all content
          * open marker present       -> reasoning starts after `<|open|>think<|sep|>`
          * open marker absent but a   close marker exists (gen-prefix consumed
            the open) -> reasoning starts at offset 0
          * response closer before any think exit -> the think channel was
            skipped; the text before it is the reply, all content
          * response opener before any think exit, given *token_ids* ->
            reasoning ends there
          * neither marker present     -> truncated reasoning after a consumed
            generation prefix
        `rest` is whatever follows the close marker, fed on to the tool parser.
        """
        if not self._thinking_enabled:
            return None, self._content_after_reasoning(model_output, request)

        m_open = self._think_open_re.search(model_output)
        # reasoning content begins right after think-open (or at start if the
        # open marker was already consumed as a generation prefix)
        content_start = m_open.end() if m_open is not None else 0

        if token_ids:
            idx, kind = self._first_think_exit(token_ids)
            m_exit = (
                self._exit_text_match(model_output, token_ids, idx, kind)
                if kind in (_RESPONSE_OPEN, _RESPONSE_CLOSE)
                else None
            )
            if m_exit is not None and kind == _RESPONSE_OPEN:
                reasoning = model_output[content_start : m_exit.start()]
                rest = model_output[m_exit.start() :]
                return reasoning or None, self._content_after_reasoning(rest, request)
            skipped = m_exit is not None
        else:
            skipped = self._skipped_think_channel(model_output, content_start)
        if skipped:
            # The model wrote its reply into the think channel; it is content.
            return None, self._content_after_reasoning(
                model_output[content_start:], request
            )

        m_close = self._think_close_re.search(model_output, content_start)
        if (
            m_open is None
            and m_close is None
            and self._response_open_re.search(model_output) is not None
        ):
            return None, self._content_after_reasoning(model_output, request)
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
          chunks: `<|open|>` / `think` / `<|sep|>reasoning`
          current text after chunk 1: `<|open|>` -> emit `""`
          current text after chunk 2: `<|open|>think` -> emit `""`
          current text after chunk 3: `<|open|>think<|sep|>reasoning`
          -> emit `reasoning`

        Example 2: split think-close marker.
          chunks: `reasoning` / `<|close|>` / `think<|sep|>...`
          after `<|close|>`, the suffix is only a partial close marker, so the
          sendable reasoning is still `reasoning` and the delta is empty. Once
          `think<|sep|>` arrives, the close branch hands the following response
          or tools text to the downstream parser.
        """
        m_open = self._think_open_re.search(text)
        if m_open is not None:
            text = text[m_open.end() :]
        overlap = 0
        # The response markers too, so no fragment of one leaks into reasoning
        # before it is recognised as leaving the think channel.
        for marker in (
            self._think_open,
            self._think_close,
            self._response_open,
            self._response_close,
        ):
            max_check = min(len(marker) - 1, len(text))
            for n in range(max_check, 0, -1):
                if text.endswith(marker[:n]):
                    overlap = max(overlap, n)
                    break
        return text[:-overlap] if overlap else text

    def _content_ready_to_emit(self, text: str) -> str:
        """Return the content prefix that is safe to stream now.

        Mirrors `_reasoning_text_ready_to_emit` but for the post-reasoning
        content phase.  Strips the `<|open|>response<|sep|>` prefix, holds back
        any partial marker suffix, and removes complete
        `<|close|>response<|sep|>` / `<|close|>message<|sep|>` markers.
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
        reasoning parser handles `<|open|>response<|sep|>` /
        `<|close|>response<|sep|>` / `<|close|>message<|sep|>` stripping itself.

        Works from accumulated text (`previous_text` / `current_text`
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
        if self._exited or self._think_close_re.search(previous_text):
            return DeltaMessage(content=delta_text)

        # The model left the think channel without closing it.
        # `is_reasoning_end_streaming` sees the same marker this step, so the
        # caller switches to content here and takes the returned content as
        # its first; the response wrappers are stripped downstream as usual.
        if self._exit_kind == -1:
            carry = max(map(len, self._exit_ids)) - 1
            self._exit_index, self._exit_kind = self._first_think_exit(
                current_token_ids, max(0, self._scanned - carry)
            )
            self._scanned = len(current_token_ids)
        kind = self._exit_kind
        m_exit = (
            self._exit_text_match(
                current_text, current_token_ids, self._exit_index, kind
            )
            if kind in (_RESPONSE_OPEN, _RESPONSE_CLOSE)
            else None
        )
        if m_exit is not None:
            self._exited = True
            self._last_streaming_delta_token_ids = tuple(delta_token_ids)
            self._last_streaming_content_token_ids = self._extract_content_ids(
                list(current_token_ids)
            )
            m_open = self._think_open_re.search(current_text)
            r_start = m_open.end() if m_open is not None else 0
            if kind == _RESPONSE_CLOSE:
                # What streamed as reasoning so far was the reply. It cannot be
                # recalled, so it is replayed as content (the copy already sent
                # stays in reasoning) -- sgl-project/sglang#37743.
                return DeltaMessage(content=current_text[r_start:] or None)
            reasoning = current_text[r_start : m_exit.start()]
            already_sent = self._reasoning_text_ready_to_emit(previous_text)
            if reasoning.startswith(already_sent):
                reasoning = reasoning[len(already_sent) :]
            return DeltaMessage(
                reasoning=reasoning or None,
                content=current_text[m_exit.start() :] or None,
            )

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
