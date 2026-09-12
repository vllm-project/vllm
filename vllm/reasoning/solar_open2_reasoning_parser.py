# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class SolarOpen2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Solar Open2 models.

    Reasoning is delimited by the ``<|think:start|>`` and ``<|think:end|>``
    special tokens. The chat template always prefills ``<|think:start|>``
    into the generation prompt, so the model output itself only ever
    contains ``<|think:end|>`` and the parser just splits around it.

    Whether the prefilled block is left *open* depends on
    ``reasoning_effort`` (``_OPEN_REASONING_EFFORTS``). Output without a
    ``<|think:end|>`` is therefore reasoning truncated mid-block when the
    effort left the block open, and plain content when the prompt already
    closed it.

    Streaming works in string space with a suffix-only hold-back, because
    vLLM's incremental decoder can split the bytes of a single special
    token across delta boundaries.

    Two things about the reasoning-end signal are non-obvious:

    - ``is_reasoning_end`` anchors on the *last* ``<|think:start|>``. The
      template renders every prior assistant turn as
      ``<|think:start|>...<|think:end|>``, so a plain containment check
      reports "ended" for any multi-turn prompt.
    - While a stream is in flight it reports the text-level state instead.
      The server stops routing deltas through this parser once the flag
      flips, so flipping while partial ``<|think:end|>`` bytes are still
      held back would leak them into the content channel.
    """

    THINK_START = "<|think:start|>"
    THINK_END = "<|think:end|>"

    # Tool-call sentinels — must stay in sync with ``SolarOpen2ToolParser``.
    # Used only to *recognize* complete embedded tool-call blocks inside
    # reasoning; actually parsing them remains the tool parser's job.
    TOOL_CALL_START = "<|tool_call:start|>"
    TOOL_CALL_END = "<|tool_call:end|>"

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        # Server-side --default-chat-template-kwargs reach the parser via the
        # constructor (same plumbing Qwen3Parser uses); a per-request effort
        # overrides this in _effective_reasoning_effort.
        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        self._default_reasoning_effort = chat_kwargs.get("reasoning_effort")

        # Both delimiters are single special tokens, so the structured-output
        # path can answer "has reasoning ended?" from token ids alone instead
        # of detokenizing the whole sequence on every scheduler step. The
        # scan stops at whichever delimiter it meets first, so it needs
        # *both* ids; a tokenizer that exposes only one of them keeps the
        # text path.
        vocab = self.vocab
        self.think_start_token_id: int | None = vocab.get(self.THINK_START)
        self.think_end_token_id: int | None = vocab.get(self.THINK_END)

        end_re = re.escape(self.THINK_END)
        self.reasoning_regex = re.compile(
            rf"^(?P<reasoning>.*?){end_re}(?P<content>.*)$",
            re.DOTALL,
        )

        tc_start = re.escape(self.TOOL_CALL_START)
        tc_end = re.escape(self.TOOL_CALL_END)
        # Body and function name of ``SolarOpen2ToolParser.tool_call_pattern``,
        # sans its capture groups. Tempering the body on the full call
        # sentinels — not on their shared ``<|tool_call:`` prefix, which an
        # argument value may quote — bounds every match attempt at the next
        # call boundary, and reading the name as the remainder of the start
        # sentinel's line keeps later newlines from becoming candidate name
        # terminators. Without both, rejecting a call the model never closed
        # costs a backtracking pass per argument group it holds.
        not_call = rf"(?:(?!{tc_start}|{tc_end}).)"
        not_call_nl = rf"(?:(?!{tc_start}|{tc_end})[^\n])"
        # Only blocks the tool parser will parse the same way once promoted
        # may leave the reasoning channel, so ``<|tool_call:end|>`` is the
        # sole terminator here, where the tool parser also accepts the next
        # ``<|tool_call:start|>``. That recovery reads the *following*
        # sentinel, which promotion cannot keep adjacent — blocks are excised
        # one at a time and rejoined — so a call missing its end sentinel
        # stays in reasoning, with the run it belongs to intact, rather than
        # being handed over in a form the tool parser reports as plain text.
        self.embedded_tool_call_regex = re.compile(
            rf"{tc_start}{not_call_nl}*?\n{not_call}*{tc_end}",
            re.DOTALL,
        )

        self._reset_stream()

    @property
    def reasoning_start_str(self) -> str:
        return self.THINK_START

    @property
    def reasoning_end_str(self) -> str:
        return self.THINK_END

    def _reset_stream(self) -> None:
        """Reset streaming-only state. Called on init and at the start of
        every new stream (detected via empty ``previous_text``)."""
        self._stream_buffer: str = ""
        self._stream_in_content: bool = False
        # ``True`` once we've handled at least one streaming delta in this
        # request — used to gate the streaming-aware ``is_reasoning_end``
        # behavior so non-streaming and prompt-side callers fall back to
        # the token-id check.
        self._stream_active: bool = False

    @staticmethod
    def _holdback_suffix(buf: str, sentinels: tuple[str, ...]) -> int:
        """Return the number of trailing bytes of ``buf`` that are a proper,
        non-empty prefix of any sentinel in ``sentinels`` — these bytes
        might complete into the sentinel once more data arrives and must
        stay in the buffer.

        Anchor on the last ``<`` (all Solar Open2 sentinels start with
        ``<|``) and only consider genuine *tail* prefixes; a letter that
        happens to appear inside a sentinel is NOT held back.
        """
        last_lt = buf.rfind("<")
        if last_lt == -1:
            return 0
        tail = buf[last_lt:]
        for s in sentinels:
            if len(tail) < len(s) and s.startswith(tail):
                return len(tail)
        return 0

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._stream_active:
            return self._stream_in_content
        start_id = self.think_start_token_id
        end_id = self.think_end_token_id
        if start_id is None or end_id is None:
            return self._is_reasoning_end_text(self.model_tokenizer.decode(input_ids))
        # Only the last think block matters, so stop at the first delimiter
        # found scanning backwards.
        for token_id in reversed(input_ids):
            if token_id == end_id:
                return True
            if token_id == start_id:
                return False
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        end_id = self.think_end_token_id
        if self._stream_active or end_id is None or self.think_start_token_id is None:
            return self.is_reasoning_end(input_ids)
        # ``delta_ids`` may be a one-shot iterator (the structured-output
        # manager passes an islice), so it must stay a single pass.
        return end_id in delta_ids

    def _is_reasoning_end_text(self, text: str) -> bool:
        last_start = text.rfind(self.THINK_START)
        if last_start == -1:
            return self.THINK_END in text
        return self.THINK_END in text[last_start + len(self.THINK_START) :]

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return []

    def _promote_embedded_tool_calls(
        self,
        reasoning: str | None,
        content: str | None,
    ) -> tuple[str | None, str | None]:
        """Move tool-call blocks embedded in reasoning into the content
        channel.

        The model occasionally emits complete tool calls *before*
        ``<|think:end|>``. The chat-completion server extracts reasoning
        first and the tool parser only inspects the content channel, so
        such calls would otherwise be silently dropped from the response.
        Promoted blocks are appended *after* the existing content because
        ``SolarOpen2ToolParser`` keeps only the text preceding the first
        ``<|tool_call:start|>`` as content — blocks placed in front would
        silently drop the real content.

        Only blocks the tool parser reads as a call *and* that carry their
        own ``<|tool_call:end|>`` are promoted; a sentinel merely
        *mentioned* in the thinking text, or a call the model never closed,
        stays in reasoning. The streaming path instead treats
        ``<|tool_call:start|>`` as an implicit reasoning end (see
        ``extract_reasoning_streaming``).
        """
        if not reasoning or self.TOOL_CALL_START not in reasoning:
            return reasoning, content

        blocks: list[str] = []

        def _collect(match) -> str:
            blocks.append(match.group(0))
            return ""

        remaining = self.embedded_tool_call_regex.sub(_collect, reasoning)
        if not blocks:
            return reasoning, content

        parts = [p for p in (content, "\n".join(blocks)) if p]
        # Removing a block can leave nothing but the whitespace that
        # surrounded it, which is not reasoning; the text that does survive
        # is kept verbatim.
        return (remaining if remaining.strip() else None), "\n".join(parts)

    # Efforts whose generation prompt ends with an *open* <|think:start|>.
    # None (unset) falls through to the chat template default, which is high.
    _OPEN_REASONING_EFFORTS = (None, "medium", "high", "xhigh")

    def _effective_reasoning_effort(
        self,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> str | None:
        """Mirror the rendering precedence: the protocol-level effort field
        (`reasoning_effort` on chat requests, `reasoning.effort` on Responses
        requests) wins over request chat_template_kwargs (merge_kwargs drops
        unset values), which in turn win over the server's default
        chat_template_kwargs."""
        if (effort := getattr(request, "reasoning_effort", None)) is not None:
            return effort
        reasoning_cfg = getattr(request, "reasoning", None)
        if (
            reasoning_cfg is not None
            and (effort := getattr(reasoning_cfg, "effort", None)) is not None
        ):
            return effort
        kwargs = getattr(request, "chat_template_kwargs", None)
        if kwargs and (effort := kwargs.get("reasoning_effort")) is not None:
            return effort
        return self._default_reasoning_effort

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        match = self.reasoning_regex.match(model_output)
        if match:
            reasoning = match.group("reasoning") or None
            content = match.group("content") or None
            return self._promote_embedded_tool_calls(reasoning, content)

        # No <|think:end|> in the output. Two prompts can lead here:
        # - closed efforts (every tier outside the open set): the template
        #   prefilled the *closed* think pair, so the whole output is content.
        # - open efforts (medium/high/xhigh and the unset default, high): the
        #   prompt ends with an *open* ``<|think:start|>``, so a marker-less
        #   output means generation was truncated mid-think — everything is
        #   reasoning, except complete embedded tool-call blocks, which are
        #   still promoted so they are not lost.
        effort = self._effective_reasoning_effort(request)
        if effort not in self._OPEN_REASONING_EFFORTS:
            return None, model_output
        return self._promote_embedded_tool_calls(model_output or None, None)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Fresh stream — reset streaming state. Production builds a parser
        # per request, but callers may reuse an instance across streams.
        if not previous_text:
            self._reset_stream()
        self._stream_active = True

        if delta_text:
            self._stream_buffer += delta_text

        # After the reasoning block closes, the stream is all content.
        if self._stream_in_content:
            if not self._stream_buffer:
                return None
            out = self._stream_buffer
            self._stream_buffer = ""
            return DeltaMessage(content=out)

        # REASONING state. Emit reasoning up to — but not including — the
        # earliest of ``<|think:end|>`` (consumed: it is a pure delimiter)
        # or ``<|tool_call:start|>`` (kept: the sentinel and everything
        # after it belong to the content channel, where the serving layer
        # feeds them through the tool parser). The latter treats a tool
        # call emitted inside the think block as an *implicit* reasoning
        # end — the streaming counterpart of ``_promote_embedded_tool_calls``
        # and the same convention the Qwen3 parser uses. If neither
        # sentinel is in the buffer yet, emit everything except trailing
        # bytes that could be the start of either.
        end_idx = self._stream_buffer.find(self.THINK_END)
        tc_idx = self._stream_buffer.find(self.TOOL_CALL_START)
        if end_idx >= 0 and (tc_idx < 0 or end_idx < tc_idx):
            reasoning_out = self._stream_buffer[:end_idx]
            content_out = self._stream_buffer[end_idx + len(self.THINK_END) :]
        elif tc_idx >= 0:
            reasoning_out = self._stream_buffer[:tc_idx]
            content_out = self._stream_buffer[tc_idx:]
        else:
            reasoning_out = content_out = ""
        if end_idx >= 0 or tc_idx >= 0:
            self._stream_buffer = ""
            self._stream_in_content = True
            # If the same chunk carried both the sentinel and some content
            # that follows it, emit both on one DeltaMessage — the OpenAI
            # streaming schema allows ``reasoning`` and ``content`` to be
            # set on the same delta, and otherwise the trailing content
            # would be stuck in the buffer if no further deltas arrive
            # (single-chunk / large-chunk synthetic inputs, final token
            # that happens to close reasoning + emit content together).
            if not reasoning_out and not content_out:
                return None
            return DeltaMessage(
                reasoning=reasoning_out or None,
                content=content_out or None,
            )

        hb = self._holdback_suffix(
            self._stream_buffer, (self.THINK_END, self.TOOL_CALL_START)
        )
        if hb == len(self._stream_buffer):
            # Every byte currently in the buffer could still become the
            # end-tag prefix. Wait.
            return None
        reasoning_out = self._stream_buffer[: len(self._stream_buffer) - hb]
        self._stream_buffer = self._stream_buffer[len(self._stream_buffer) - hb :]
        if not reasoning_out:
            return None
        return DeltaMessage(reasoning=reasoning_out)
