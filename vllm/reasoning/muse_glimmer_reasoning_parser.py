# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-content parser for MuseGlimmer.
Port of the ``reasoning_content`` rule from the HuggingFace MuseGlimmer
``MUSE_GLIMMER_RESPONSE_SCHEMA`` (synced with internal master). MuseGlimmer emits
chain-of-thought in ``to=self`` channels delimited by ``<|message|>`` ... ``<|eom|>``:
    to=self<|message|>...reasoning...<|eom|>
A turn may contain several ``to=self`` blocks interleaved with tool calls, and a
tool call or final answer follows in its own channel.
Because MuseGlimmer's framing markers (``<|message|>``, ``<|eom|>``) are not guaranteed
to be single vocab tokens across every checkpoint's tokenizer, this parser works
on the decoded text with regexes rather than the single start/end-token base class.
Usage: ``--reasoning-parser muse_glimmer``
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

_EOM = "<|eom|>"
_EOT = "<|eot|>"
_FUNCTION_CALLS_OPEN = "<atem:function_calls>"
_REASONING_OPEN = "to=self<|message|>"
_ASSISTANT_TURN_OPEN = "<|start|>assistant"
# A channel header: ``to=<recipient><|message|>`` where recipient is ``self``
# (reasoning), ``user`` (final answer) or ``<tool>[.<fn>]`` (tool call).
_CHANNEL_HEADER_RE = re.compile(r"to=(?P<recipient>[^\s<]+)<\|message\|>")
_HEADER_PAT = r"to=[^\s<]+<\|message\|>"
# Collapse the gap between reasoning blocks so multiple to=self spans join.
_COLLAPSE_RE = re.compile(
    r"<\|eom\|>(?:(?!to=self<\|message\|>).)*?to=self<\|message\|>", re.DOTALL
)
_REASONING_RE = re.compile(r"to=self<\|message\|>(.*?)<\|eom\|>", re.DOTALL)
_CONTENT_RE = re.compile(
    r"to=user<\|message\|>(.*?)(?=<\|eot\|>|<\|eom\|>|$)", re.DOTALL
)
# Strip a CLOSED reasoning span (header .. <|eom|>).
_STRIP_REASONING_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?to=self<\|message\|>.*?<\|eom\|>", re.DOTALL
)
# An UNTERMINATED trailing reasoning span. The model sometimes leaves the
# analysis channel WITHOUT emitting <|eom|>, writing a bare
# ``to=<tool><|message|>`` header instead (observed deterministically for a call
# with EMPTY arguments on a tool that has optional parameters; reproduced on
# other engines too, so it is a model-side defect, not engine-specific).
#
# These two patterns MUST therefore stop at the next channel header rather than
# running to end-of-text. An unbounded ``...$`` version consumes the real tool
# call along with the reasoning: `is_reasoning_end` then never fires, the parser
# never leaves the reasoning phase, the tool parser is never invoked, and the
# entire generation is dropped (empty reasoning, empty content, no tool call).
_STRIP_OPEN_REASONING_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?to=self<\|message\|>"
    r"(?:(?!<\|eom\|>)(?!" + _HEADER_PAT + r").)*"
    r"(?=" + _HEADER_PAT + r"|$)",
    re.DOTALL,
)
_OPEN_REASONING_RE = re.compile(
    r"to=self<\|message\|>((?:(?!<\|eom\|>)(?!" + _HEADER_PAT + r").)*)"
    r"(?=" + _HEADER_PAT + r"|$)",
    re.DOTALL,
)
# Markers whose PREFIX could appear at the tail of an OPEN (still-streaming) body.
_HOLDBACK_MARKERS = (_EOM, _EOT, "<|start|>", "<|message|>")
# A trailing fragment that could still grow into a channel header (" t", " to",
# " to=", " to=skill"). Without this the recipient name leaks into reasoning and
# then has to be un-emitted once ``<|message|>`` arrives.
_OPEN_TAIL_HEADER_RE = re.compile(r"[\s](?:t|to|to=[^\s<]*)$")


def _current_assistant_turn(text: str) -> str:
    """Return only the text generated in the current assistant turn.
    ``is_reasoning_end`` is evaluated on the PROMPT token-ids at stream start,
    and an MuseGlimmer prompt legitimately contains ATEM markers (``render_tool_defs``
    writes a literal ``<atem:function_calls>`` example into the system message,
    and prior assistant turns may carry real tool calls). Anchoring on the last
    channel-open keeps prompt text from deciding the phase.
    """
    idx = text.rfind(_ASSISTANT_TURN_OPEN)
    return text[idx + len(_ASSISTANT_TURN_OPEN) :] if idx != -1 else text


def _trim_open_body(body: str) -> str:
    """Hold back any tail of a still-growing body that could still be framing.
    Iterated to a fixpoint because the two cases compose: `" to=skill<"` needs
    the partial-marker trim (``<``) before the partial-header trim can see
    `" to=skill"`. Trimming only once leaks the recipient name as reasoning.
    """
    while True:
        trimmed = body
        for marker in _HOLDBACK_MARKERS:
            for k in range(min(len(marker) - 1, len(trimmed)), 0, -1):
                if trimmed.endswith(marker[:k]):
                    trimmed = trimmed[:-k]
                    break
            else:
                continue
            break
        header_tail = _OPEN_TAIL_HEADER_RE.search(trimmed)
        if header_tail is not None:
            trimmed = trimmed[: header_tail.start()]
        if trimmed == body:
            return body
        body = trimmed


class MuseGlimmerReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        # Cursors over what was ACTUALLY emitted. Diffing a freshly reclassified
        # `previous_text` is unsafe: a classified body legitimately shrinks when
        # a partial header becomes recognisable, and diffing against the shrunken
        # value re-emits text that already went out.
        self._emitted_reasoning: str = ""
        self._emitted_content: str = ""
        self._tool_handoff_done: bool = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Preserve MuseGlimmer's ATEM framing tokens in the decoded output.
        vLLM's serving default is ``skip_special_tokens=True``, which strips
        ``<|start|>`` / ``<|message|>`` / ``<|eom|>`` / ``<|eot|>`` before the
        parsers run, collapsing reasoning into content and breaking channel
        scoping. Unlike the base tool-parser hook we do NOT touch
        ``structured_outputs`` -- MuseGlimmer emits native ATEM, not JSON.
        """
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Whether the model has left reasoning and opened a TOOL channel.
        A ``to=user`` answer is NOT a reason to leave the reasoning phase -- this
        parser surfaces that content itself. Only a real tool channel switches
        the ``DelegatingParser`` phase machine over to the tool parser.
        Both closed and unterminated reasoning spans are stripped before the
        check, so an ``<atem:invoke>`` the model merely echoes inside its CoT
        never flips the phase.
        """
        try:
            text = self.model_tokenizer.decode(input_ids)
        except Exception:
            return False
        remainder = self._tool_channel_remainder(text)
        return _FUNCTION_CALLS_OPEN in remainder or "<atem:invoke" in remainder

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.is_reasoning_end(list(input_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Content-id slicing is unreliable for multi-token markers; the serving
        # path uses extract_reasoning() for the final split.
        return []

    @classmethod
    def _scoped_turn(cls, text: str) -> str:
        """Current assistant turn with reasoning spans removed."""
        scoped = _current_assistant_turn(text)
        scoped = _STRIP_REASONING_RE.sub("", scoped)
        return _STRIP_OPEN_REASONING_RE.sub("", scoped)

    @classmethod
    def _tool_channel_remainder(cls, text: str) -> str:
        """Text from the first tool-channel header onward, framing INCLUDED.
        ``DelegatingParser.parse_delta`` rebuilds ``current_text`` from whatever
        this parser returns as ``.content`` on the transition delta and commits
        it; anything not returned is destroyed. It must start AT the
        ``to=<name><|message|>`` header -- handing over the text after the header
        loses the recipient, and the tool parser then sees a bare ``<|message|>``,
        classifies it as the content channel, and leaks the ATEM markup.
        """
        scoped = cls._scoped_turn(text)
        for match in _CHANNEL_HEADER_RE.finditer(scoped):
            if match.group("recipient") not in ("self", "user"):
                return scoped[match.start() :]
        return ""

    @staticmethod
    def _classify_bodies(text: str) -> tuple[str, str]:
        """Split ``text`` into (reasoning_body, content_body), channel-aware.
        Framing markers and tool channels contribute nothing -- the tool parser
        owns those. A body ends at ``<|eom|>`` / ``<|eot|>``, at the next channel
        header, or at end-of-text (an OPEN body, which is held back).
        """
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        pos = 0
        n = len(text)
        while pos < n:
            match = _CHANNEL_HEADER_RE.search(text, pos)
            if not match:
                break
            recipient = match.group("recipient")
            body_start = match.end()
            eom = text.find(_EOM, body_start)
            eot = text.find(_EOT, body_start)
            terminators = [p for p in (eom, eot) if p != -1]
            next_header = _CHANNEL_HEADER_RE.search(text, body_start)
            if next_header is not None:
                terminators.append(next_header.start())
            body_end = min(terminators) if terminators else n
            body = text[body_start:body_end]
            if not terminators:
                body = _trim_open_body(body)
            if recipient == "self":
                reasoning_parts.append(body)
            elif (
                # Never surface tool XML echoed into a user channel.
                recipient == "user"
                and _FUNCTION_CALLS_OPEN not in body
                and "<atem:invoke" not in body
            ):
                content_parts.append(body)
            if terminators and body_end in (eom, eot):
                pos = body_end + len(_EOM if body_end == eom else _EOT)
            else:
                pos = body_end
        return "".join(reasoning_parts), "".join(content_parts)

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        """Promote un-surfaced content when the stream ends mid-reasoning.
        ``DelegatingParser.finalize_generation`` calls this when
        ``reasoning_ended`` is still False. Returns only the channel-classified
        ``to=user`` body, and only the portion not already streamed.
        """
        _, content_body = self._classify_bodies(previous_text)
        remainder = content_body[len(self._emitted_content) :]
        return remainder or None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        collapsed = _COLLAPSE_RE.sub("\n", model_output)
        matches = _REASONING_RE.findall(collapsed)
        reasoning = "\n".join(matches) if matches else None
        # Truncation fallback: generation stopped inside a to=self block, so
        # there is no closing <|eom|>. Bounded at the next channel header so a
        # real tool call that follows a header-less channel switch is not
        # absorbed into the reasoning field.
        open_match = _OPEN_REASONING_RE.search(model_output)
        if open_match and open_match.group(1):
            partial = open_match.group(1)
            reasoning = f"{reasoning}\n{partial}" if reasoning else partial
        # Content is everything that is not a reasoning block. In a
        # reasoning+tool-call turn there is no to=user answer, but the tool
        # channels MUST be forwarded -- the unified parser runs the tool parser
        # on this returned `content`, not on the original model_output.
        remainder = _STRIP_REASONING_RE.sub("", model_output)
        remainder = _STRIP_OPEN_REASONING_RE.sub("", remainder)
        if "<atem:invoke" in remainder or _FUNCTION_CALLS_OPEN in remainder:
            return reasoning, (remainder or None)
        content_match = _CONTENT_RE.search(model_output)
        if content_match:
            content = content_match.group(1) or None
        elif _REASONING_OPEN in model_output:
            content = None
        else:
            content = model_output or None
            reasoning = None
        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Channel-aware streaming split of reasoning vs content.
        Classifies the full ``current_text`` and emits only what has not been
        emitted yet, so no framing token is ever surfaced and a delta straddling
        a channel boundary only contributes the portion inside a real body.
        """
        curr_reason, curr_content = self._classify_bodies(current_text)
        reasoning_delta = ""
        if curr_reason.startswith(self._emitted_reasoning) and len(curr_reason) > len(
            self._emitted_reasoning
        ):
            reasoning_delta = curr_reason[len(self._emitted_reasoning) :]
            self._emitted_reasoning = curr_reason
        content_delta = ""
        if curr_content.startswith(self._emitted_content) and len(curr_content) > len(
            self._emitted_content
        ):
            content_delta = curr_content[len(self._emitted_content) :]
            self._emitted_content = curr_content
        # Hand the tool channel to the tool parser exactly once, starting at its
        # header. parse_delta discards anything not returned here.
        #
        # This MUST fire on the same delta where is_reasoning_end() flips, i.e.
        # only once the tool channel actually contains ATEM. Emitting it earlier
        # -- when only the bare `to=<name><|message|>` header has arrived -- keeps
        # the parser in the reasoning phase, so parse_delta never replaces this
        # DeltaMessage with the tool parser's and the header is delivered to the
        # client as visible content.
        handoff = ""
        if not self._tool_handoff_done:
            remainder = self._tool_channel_remainder(current_text)
            if _FUNCTION_CALLS_OPEN in remainder or "<atem:invoke" in remainder:
                handoff = remainder
                self._tool_handoff_done = True
        if handoff:
            return DeltaMessage(reasoning=reasoning_delta or None, content=handoff)
        if reasoning_delta and content_delta:
            return DeltaMessage(reasoning=reasoning_delta, content=content_delta)
        if reasoning_delta:
            return DeltaMessage(reasoning=reasoning_delta)
        if content_delta:
            return DeltaMessage(content=content_delta)
        return None
