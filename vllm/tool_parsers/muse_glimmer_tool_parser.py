# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ATEM tool-call parser for MuseGlimmer.

Faithful port of the MuseGlimmer ``response_schema`` tool-call contract from the
HuggingFace MuseGlimmer export (``convert_muse_glimmer_weights_to_hf.py``:
``MUSE_GLIMMER_RESPONSE_SCHEMA``).

MuseGlimmer emits tool calls in an XML-ish ATEM format inside channel-scoped messages:

    <|start|>assistant to=self<|message|>...reasoning...<|eom|>
    <|start|>assistant to=<tool>.<fn><|message|>
        <atem:function_calls>
        <atem:invoke name="tool.fn">
        <atem:parameter name="arg">value</atem:parameter>
        </atem:invoke>
        </atem:function_calls><|eom|>            # non-final call
    <|start|>assistant to=user<|message|>...final answer...<|eot|>

Channel scoping is essential: an ``<atem:invoke>`` echoed inside a ``to=self``
reasoning block or a ``to=user`` final answer must NOT be parsed as a real tool
call.

Rather than *subtracting* reasoning/answer spans with regex substitutions (the
approach the HF ``response_schema`` uses, which is safe only on a complete,
well-formed turn), this parser *segments* the output into messages and then
*selects* the tool-channel bodies. On a complete turn the two are equivalent;
on a truncated or damaged turn, subtraction can delete a valid tool call
(an unterminated ``to=self`` block makes the non-greedy strip run to the next
``<|eom|>``, which belongs to the tool-call message) whereas selection cannot.

Usage: ``--enable-auto-tool-choice --tool-call-parser muse_glimmer``
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

import regex as re
from openai.types.responses import ToolChoiceFunction
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)

logger = init_logger(__name__)

# --- Message framing -------------------------------------------------------
# An assistant message header. All three parts are optional except the
# <|message|> terminator:
#   "<|start|>assistant to=get_weather<|message|>"  -- after an <|eom|> boundary
#   " to=self<|message|>"                           -- first message of a turn
#                                                      (the prompt already ended
#                                                      with "<|start|>assistant")
#   "<|message|>"                                   -- bare recipient (public CoT
#                                                      / untagged content)
_MSG_HEADER_RE = re.compile(
    r"(?:<\|start\|>\s*assistant)?[^\S\n]*(?:to=(?P<rcpt>[A-Za-z0-9_.\-]+))?<\|message\|>"
)
_MSG_END_RE = re.compile(r"<\|eom\|>|<\|eot\|>")

# Recipients whose bodies are NOT tool calls.
_REASONING_RECIPIENT = "self"
_USER_RECIPIENT = "user"

# Structural markers that must never reach the client. A streamed body is held
# back by up to len(marker)-1 characters so a marker split across two chunks is
# not emitted as content.
_STRUCTURAL_MARKERS = ("<|eom|>", "<|eot|>", "<|start|>", "<|message|>")
_MAX_MARKER_LEN = max(len(m) for m in _STRUCTURAL_MARKERS)
# A trailing " to=NAME" that could still grow into a bare message header (the
# first message of a turn has no <|start|> prefix -- the prompt ends with
# "<|start|>assistant", so the model's first emitted text is " to=self<|message|>").
_OPEN_TAIL_TO_RE = re.compile(r"[^\S\n]+to=[A-Za-z0-9_.\-]*$")

# --- Tool-call extraction (unchanged from MUSE_GLIMMER_RESPONSE_SCHEMA) -------------
_INVOKE_RE = re.compile(r"(<atem:invoke\b.*?</atem:invoke>)", re.DOTALL)
_NAME_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)"')
_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)
_FUNCTION_CALLS_OPEN = "<atem:function_calls>"


def _decode_value(raw: str):
    """JSON-decode a parameter value when possible, else keep the raw string.

    Mirrors the schema's ``x-parser: json`` with ``allow_non_json: True``.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _iter_messages(text: str) -> Iterator[tuple[str | None, str, bool]]:
    """Segment *text* into assistant messages.

    Yields ``(recipient, body, closed)`` per message, where ``recipient`` is
    ``None`` for a bare ``<|message|>`` header and ``closed`` is False for a
    message that has not (yet) seen ``<|eom|>`` / ``<|eot|>``.

    A message is also terminated by the start of the NEXT header. Without that,
    a reasoning block whose ``<|eom|>`` is missing (truncation, or a chunk
    dropped at the reasoning -> tool transition) would absorb the tool-call
    message that follows it and the call would be lost -- the same defect the
    subtractive regexes have.
    """
    pos = 0
    while pos < len(text):
        header = _MSG_HEADER_RE.search(text, pos)
        if header is None:
            return
        body_start = header.end()
        end = _MSG_END_RE.search(text, body_start)
        nxt = _MSG_HEADER_RE.search(text, body_start)
        body_end = end.start() if end is not None else len(text)
        closed = end is not None
        if nxt is not None and nxt.start() < body_end:
            body_end = nxt.start()
            closed = False
            next_pos = nxt.start()
        else:
            next_pos = end.end() if end is not None else len(text)
        body = text[body_start:body_end]
        # A body can never legitimately contain <|start|>. Seeing one means the
        # next header is only partially generated (its <|message|> has not
        # arrived), so the regex above could not recognise it yet. Cut there,
        # otherwise the streamed body would grow to include the next header and
        # then shrink back once it completes.
        start_tok = body.find("<|start|>")
        if start_tok != -1:
            body = body[:start_tok]
            closed = False
        yield header.group("rcpt"), body, closed
        pos = next_pos


def _trailing_partial_marker_len(text: str) -> int:
    """Length of the longest suffix of *text* that prefixes a structural marker."""
    max_overlap = min(len(text), _MAX_MARKER_LEN - 1)
    for overlap in range(max_overlap, 0, -1):
        suffix = text[-overlap:]
        if any(marker.startswith(suffix) for marker in _STRUCTURAL_MARKERS):
            return overlap
    return 0


def _safe_open_body(body: str) -> str:
    """Trim the tail of a still-growing body to what is safe to emit now.

    Holds back anything that could still turn out to be structural, so the
    emitted prefix only ever grows. Chunks under speculative decoding are large
    enough that markers routinely straddle them.
    """
    tail_to = _OPEN_TAIL_TO_RE.search(body)
    if tail_to is not None:
        return body[: tail_to.start()]
    partial = _trailing_partial_marker_len(body)
    return body[: len(body) - partial] if partial else body


class MuseGlimmerToolParser(ToolParser):
    # MuseGlimmer emits ATEM markup around tool-call arguments. The generic
    # named/required tool_choice path in vllm/parser/abstract_parser.py assigns
    # the raw model text straight to FunctionCall.arguments, which leaks that
    # framing and yields invalid JSON; the "required" branch then silently
    # swallows the ValidationError and returns tool_calls=null. Opting out
    # routes named/required through extract_tool_calls /
    # extract_tool_calls_streaming -- the same path "auto" already uses.
    supports_required_and_named = False

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools)
        # Streaming cursors. vLLM constructs one ToolParser per request, so
        # instance state is per-stream.
        self._streamed_content_len: int = 0
        self._streamed_reasoning_len: int = 0
        self._emitted_tool_calls: int = 0

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Force special tokens through, and keep JSON guided decoding off.

        ``skip_special_tokens`` defaults to True on both ChatCompletionRequest
        and ResponsesRequest. Every rule in this parser keys off ``<|message|>``
        / ``<|eom|>`` / ``<|eot|>`` / ``<|start|>``, so with the default the
        channel framing is stripped before we see it: message segmentation
        finds nothing, reasoning-channel invokes are indistinguishable from real
        ones, and the raw ATEM markup falls through to the client as content.
        Set it unconditionally and FIRST -- not only for tools requests, and not
        relying on the reasoning parser to have set it.

        For required/named ``tool_choice`` the base hook installs a JSON schema
        constraint (ToolParser.adjust_request -> get_json_schema_from_tools).
        MuseGlimmer emits ATEM XML, so under that constraint it writes JSON *inside*
        the tool channel -- ``<atem:function_calls>[{"name": ...``  -- with no
        ``<atem:invoke>`` for this parser to find. Skip the base hook so those
        choices decode natively, the same as "auto".
        """
        request.skip_special_tokens = False

        tool_choice = getattr(request, "tool_choice", None)
        if request.tools and (
            tool_choice == "required"
            or isinstance(
                tool_choice, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            )
        ):
            return request
        return super().adjust_request(request)

    # ---------------- channel selection ----------------

    @classmethod
    def _tool_channel_text(cls, text: str) -> str:
        """Concatenate the bodies of messages addressed to a tool.

        Falls back to the whole text when no message header is present at all --
        that means the framing never reached us (``skip_special_tokens`` was on,
        or the chunk carrying the header was dropped upstream), and scanning
        everything is strictly better than returning nothing.
        """
        bodies = [
            body
            for rcpt, body, _closed in _iter_messages(text)
            if rcpt is not None
            and rcpt != _REASONING_RECIPIENT
            and rcpt != _USER_RECIPIENT
        ]
        if bodies:
            return "\n".join(bodies)
        if _MSG_HEADER_RE.search(text) is None and (
            _FUNCTION_CALLS_OPEN in text or "<atem:invoke" in text
        ):
            logger.warning(
                "MuseGlimmer: ATEM markup with no channel framing; "
                "is skip_special_tokens enabled upstream?"
            )
            return text
        return ""

    @classmethod
    def _visible_channels(cls, text: str) -> tuple[str, str, bool, bool]:
        """Return ``(content, reasoning, content_open, reasoning_open)``.

        The ``*_open`` flags say whether that channel's LAST message is still
        being generated; only then must the caller hold back a partial
        structural marker. Tracking them per channel matters: a closed
        reasoning block whose text happens to end in ``<`` would otherwise stay
        permanently truncated while a later content message is open.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        content_open = False
        reasoning_open = False
        for rcpt, body, closed in _iter_messages(text):
            if rcpt == _REASONING_RECIPIENT:
                reasoning_parts.append(body)
                reasoning_open = not closed
            elif rcpt is None or rcpt == _USER_RECIPIENT:
                content_parts.append(body)
                content_open = not closed
        return (
            "".join(content_parts),
            "".join(reasoning_parts),
            content_open,
            reasoning_open,
        )

    # ---------------- tool name binding ----------------

    @staticmethod
    def _registered_names(request: ChatCompletionRequest | None) -> set[str]:
        """Names of the tools the client registered on this request."""
        names: set[str] = set()
        tools = getattr(request, "tools", None) if request is not None else None
        for t in tools or []:
            fn = getattr(t, "function", None) or t
            name = getattr(fn, "name", None)
            if name is None and isinstance(fn, dict):
                name = fn.get("name")
            if name:
                names.add(name)
        return names

    @staticmethod
    def _normalize_name(emitted: str, registered: set[str]) -> str:
        """Map an emitted ATEM invoke name back to a registered tool name.

        When a client registers a BARE name (e.g. ``get_weather``) the shipped
        chat template renders the valid recipient as ``"get_weather.*"``, and
        the model duly emits ``get_weather.get_weather``. Collapsing that
        doubled form is safe: head and tail are identical and the collapsed
        name is registered.

        Anything else is passed through unchanged. Matching on the trailing
        segment alone is NOT safe -- an emitted ``weather.get`` against a
        registered ``{calendar.get}`` has a unique leaf match and would silently
        dispatch the wrong tool.
        """
        if not registered or emitted in registered:
            return emitted
        head, sep, tail = emitted.partition(".")
        if sep and head == tail and head in registered:
            return head
        logger.warning(
            "MuseGlimmer: emitted tool name %r does not match any registered tool; "
            "passing through unchanged.",
            emitted,
        )
        return emitted

    @classmethod
    def _parse_tool_calls(
        cls, text: str, registered: set[str] | None = None
    ) -> list[ToolCall]:
        registered = registered or set()
        scoped = cls._tool_channel_text(text)
        tool_calls: list[ToolCall] = []
        for invoke in _INVOKE_RE.findall(scoped):
            name_m = _NAME_RE.search(invoke)
            if not name_m:
                continue
            name = cls._normalize_name(name_m.group(1), registered)
            args: dict = {}
            for pm in _PARAM_RE.finditer(invoke):
                args[pm.group("key")] = _decode_value(pm.group("value"))
            tool_calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(args, ensure_ascii=False),
                    )
                )
            )
        return tool_calls

    @classmethod
    def _extract_content(cls, text: str) -> str | None:
        """Return the user-facing body, or the raw text when unframed."""
        content, reasoning, _c_open, _r_open = cls._visible_channels(text)
        if content:
            return content
        # No framing at all -> the whole thing is plain content.
        if not reasoning and _MSG_HEADER_RE.search(text) is None:
            return text or None
        return None

    # ---------------- non-streaming ----------------

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        if (
            _FUNCTION_CALLS_OPEN not in model_output
            and "<atem:invoke" not in model_output
        ):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self._extract_content(model_output),
            )
        try:
            registered = self._registered_names(request)
            tool_calls = self._parse_tool_calls(model_output, registered)
            if not tool_calls:
                # A tool block was opened but no COMPLETE
                # <atem:invoke>...</atem:invoke> parsed -- typically a truncated
                # call (finish_reason='length'/abort). Log it: silently
                # returning "no tool call" here is indistinguishable from the
                # model choosing not to call one, which makes this failure mode
                # invisible in production.
                logger.warning(
                    "MuseGlimmer: tool channel opened but no complete <atem:invoke> "
                    "parsed (truncated tool call?); returning content only."
                )
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=self._extract_content(model_output),
                )
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=self._extract_content(model_output),
            )
        except Exception:
            logger.exception("Error extracting MuseGlimmer ATEM tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ---------------- streaming ----------------

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Incremental ATEM streaming for tool calls AND content.

        This parser owns every delta once reasoning has ended: in
        vllm/parser/abstract_parser.py::parse_delta the "pass through as
        content" fallback is guarded by ``not self._in_tool_call_phase(state)``,
        and ``_in_tool_call_phase`` is simply ``tool_parser is not None and
        state.reasoning_ended``. So with a tool parser loaded that fallback is
        dead code, and returning None here DISCARDS the delta. Anything we do
        not emit -- including the ``to=user`` final answer -- never reaches the
        client. Hence content is emitted here, not left to the reasoning parser.

        Tool calls are surfaced only when an ``<atem:invoke>`` block becomes
        complete: the XML is opaque until closed and MuseGlimmer parameters are not
        incremental JSON, so there is nothing meaningful to stream before then.
        """
        if not previous_text:
            # First delta of the tool phase (parse_delta resets previous_text
            # to "" when it hands the stream over). Reset the cursors.
            self._streamed_content_len = 0
            self._streamed_reasoning_len = 0
            self._emitted_tool_calls = 0

        try:
            registered = self._registered_names(request)
            calls = self._parse_tool_calls(current_text, registered)
            content, reasoning, content_open, reasoning_open = self._visible_channels(
                current_text
            )

            # Trim the tail of a channel that is still growing, so the emitted
            # prefix never shrinks between deltas.
            if content_open:
                content = _safe_open_body(content)
            if reasoning_open:
                reasoning = _safe_open_body(reasoning)

            content_delta = content[self._streamed_content_len :]
            reasoning_delta = reasoning[self._streamed_reasoning_len :]

            tool_deltas: list[DeltaToolCall] = []
            for i in range(self._emitted_tool_calls, len(calls)):
                fn = calls[i].function
                tool_deltas.append(
                    DeltaToolCall(
                        index=i,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=fn.name,
                            arguments=fn.arguments,
                        ).model_dump(exclude_none=True),
                    )
                )

            if not content_delta and not reasoning_delta and not tool_deltas:
                return None

            self._streamed_content_len = len(content)
            self._streamed_reasoning_len = len(reasoning)
            self._emitted_tool_calls = len(calls)

            message = DeltaMessage()
            if content_delta:
                message.content = content_delta
            if reasoning_delta:
                message.reasoning = reasoning_delta
            if tool_deltas:
                message.tool_calls = tool_deltas
            return message
        except Exception:
            logger.exception("Error extracting MuseGlimmer ATEM streaming tool calls.")
            return None
