# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
import re
from collections.abc import Sequence
from enum import Enum, auto
from typing import Optional

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
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
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import find_tool_properties

logger = init_logger(__name__)


class _State(Enum):
    TEXT = auto()  # Plain text / before any tool call
    TOOL_CALL_PENDING = auto()  # Saw <tool_call>; waiting for lookahead tag
    TOOL_CALL = auto()  # Between structural tags inside <tool_call>
    FUNCTION = auto()  # Between structural tags inside <function=...>
    PARAM_VALUE = auto()  # Accumulating parameter value text
    PARAM_CLOSE_PENDING = auto()  # Saw </parameter>; waiting for lookahead tag


class StreamingXMLToolCallParser:
    """
    Character-level streaming XML tool call parser implemented as a state
    machine.

    Processes input one character at a time with no assumptions about input
    chunk boundaries.  Supports the Qwen3 XML tool-call format::

        <tool_call>
        <function=FUNCTION_NAME>
        <parameter=PARAM_NAME>
        VALUE
        </parameter>
        </function>
        </tool_call>

    Both ``<function=name>`` and ``<function name="name">`` are accepted, and
    likewise for ``<parameter>``.  ``<function=…>`` and ``<parameter=…>`` are
    only meaningful inside an open ``<tool_call>`` context; they are treated as
    plain text otherwise.

    Inside a tag (the ``<…>`` region) the parser also tracks quoted strings so
    that ``<`` characters inside attribute values (e.g. ``<foo attr="a<b">``)
    do not prematurely restart tag accumulation.
    """

    # Kept for callers that reference them directly.
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    function_start_token: str = "<function="
    function_end_token: str = "</function>"
    parameter_start_token: str = "<parameter="
    parameter_end_token: str = "</parameter>"

    # Function names must start with a letter or underscore and contain only
    # alphanumerics, underscores, dots, colons, or hyphens.  This prevents
    # documentation placeholders like `<function=...>` from being parsed as
    # real function-open tags.
    _RE_FUNCTION_OPEN = re.compile(
        r'^<function(?:\s*=\s*([a-zA-Z_][a-zA-Z0-9_.:\-]*)'
        r'|(?:\s+name\s*=\s*["\x27]([^"\x27]+)["\x27]))\s*>$'
    )
    _RE_PARAMETER_OPEN = re.compile(
        r'^<parameter(?:\s*=\s*([^>\s]+)|(?:\s+name\s*=\s*["\x27]([^"\x27]+)["\x27]))\s*>$'
    )

    # Fixed-prefix set used for early tag rejection.  For variable-suffix tags
    # (those ending with "=" or " ") the fixed portion is listed here; any
    # content after that prefix is valid.  Completely fixed tags (no variable
    # portion before ">") are also listed so character-by-character matching
    # works up to the ">".
    _STRUCTURAL_TAG_PREFIXES: tuple[str, ...] = (
        "<tool_call>",
        "</tool_call>",
        "<function=",   # <function=NAME> form — variable after "="
        "<function ",   # <function name="NAME"> form — variable after space
        "</function>",
        "<parameter=",  # <parameter=NAME> form — variable after "="
        "<parameter ",  # <parameter name="NAME"> form — variable after space
        "</parameter>",
    )

    def __init__(self) -> None:
        self.tools: Optional[list[Tool]] = None
        self.reset_streaming_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_tools(self, tools: Optional[list[Tool]]) -> None:
        self.tools = tools

    def reset_streaming_state(self) -> None:
        """Reset all mutable parser state."""
        self.deltas: list[DeltaMessage] = []

        # State machine
        self._state: _State = _State.TEXT

        # True while accumulating the content of a <…> tag.
        self._in_tag: bool = False
        # Buffer for the current <…> tag (including the opening '<').
        self._tag_buf: str = ""
        # True when we're inside a quoted string within a tag attribute value.
        self._tag_in_quote: bool = False
        # The opening quote character (' or ") when _tag_in_quote is True.
        self._tag_quote_char: str = ""

        # Plain-text accumulator used in TEXT state.
        self._text_buf: str = ""
        # Raw parameter-value accumulator.
        self._param_buf: str = ""
        # Whitespace/chars buffered between a tentative </parameter> and the
        # next tag in PARAM_CLOSE_PENDING state (restored as content on
        # lookahead failure).
        self._pending_close_ws: str = ""
        # Chars buffered between a tentative <tool_call> and the next tag in
        # TOOL_CALL_PENDING state (restored as content on lookahead failure).
        self._pending_tool_call_ws: str = ""

        # Semantic context
        self.tool_call_index: int = 0
        self._call_id: Optional[str] = None
        self._function_name: Optional[str] = None
        self._param_name: Optional[str] = None
        # Names of parameters whose JSON key+value have already been emitted.
        self._params_emitted: list[str] = []
        # True once _start_function() has been called for the current tool call.
        # Guards _end_tool_call() so it never emits a terminator delta for a
        # call that never had a function (which would produce name=None/args=''
        # ghosts when the stream is aborted before the function tag is complete).
        self._function_ever_started: bool = False

    def parse_single_streaming_chunks(self, chunk: str) -> DeltaMessage:
        """
        Feed *chunk* through the parser and return a single merged
        :class:`DeltaMessage` representing all output produced by this chunk.
        """
        initial_count = len(self.deltas)

        for ch in chunk:
            self._feed(ch)

        # Emit any buffered plain text that has accumulated outside a tool call.
        if self._state == _State.TEXT and not self._in_tag and self._text_buf:
            self._emit(DeltaMessage(content=self._text_buf))
            self._text_buf = ""

        return self._merge_deltas(self.deltas[initial_count:])

    def finalize(self) -> DeltaMessage:
        """
        Close any open structural elements and return a delta for remaining
        output.  Call this once the full model output has been fed in
        (non-streaming path).
        """
        initial_count = len(self.deltas)
        if self._state == _State.TOOL_CALL_PENDING:
            self._text_buf += "<tool_call>" + self._pending_tool_call_ws
            self._pending_tool_call_ws = ""
            self._state = _State.TEXT
        if self._param_name is not None:
            self._end_parameter()
        if self._function_name is not None:
            self._end_function()
        if self._call_id is not None:
            self._end_tool_call()
        self._flush_text_as_content()
        return self._merge_deltas(self.deltas[initial_count:])

    def collect_all(self) -> DeltaMessage:
        """
        Return a merged view of every delta emitted so far (read-only).

        Tool calls that never acquired a function name (e.g. streams aborted
        before the ``<function=…>`` tag was complete) are filtered out so the
        caller never sees ``name=None / arguments=''`` ghost entries.
        """
        result = self._merge_deltas(self.deltas)
        if result.tool_calls:
            valid = [tc for tc in result.tool_calls if tc.function and tc.function.name]
            if valid:
                result = DeltaMessage(
                    content=result.content,
                    tool_calls=valid,
                    reasoning=result.reasoning,
                )
            else:
                result = DeltaMessage(content=result.content, reasoning=result.reasoning)
        return result

    # ------------------------------------------------------------------
    # Core character feed
    # ------------------------------------------------------------------

    def _feed(self, ch: str) -> None:
        if self._in_tag:
            self._feed_in_tag(ch)
        elif self._state == _State.PARAM_VALUE:
            self._feed_param_value(ch)
        elif self._state == _State.PARAM_CLOSE_PENDING:
            self._feed_param_close_pending(ch)
        elif self._state == _State.TOOL_CALL_PENDING:
            self._feed_tool_call_pending(ch)
        else:
            self._feed_context(ch)

    def _tag_prefix_is_viable(self) -> bool:
        """Return True if _tag_buf is still a valid prefix of some structural tag.

        Called after each character is appended inside a tag.  When it returns
        False the caller should flush the buffer immediately rather than waiting
        for the closing ">".  This avoids long buffering delays for sequences
        like "<--", "<!", "< ", "<br>", etc. that can never form a structural
        tag.

        For variable-suffix tags (those whose fixed prefix ends with "=" or " ")
        any content after the fixed prefix is accepted as part of the name, so
        once the buffer has grown past the fixed prefix the check always
        succeeds.
        """
        buf = self._tag_buf
        n = len(buf)
        if n <= 1:
            return True
        for tag in self._STRUCTURAL_TAG_PREFIXES:
            tlen = len(tag)
            if n <= tlen:
                if tag[:n] == buf:
                    return True
            else:
                # buf is longer than the fixed prefix: viable only for
                # variable-suffix tags where the fixed prefix matches exactly.
                if buf[:tlen] == tag:
                    return True
        return False

    def _feed_in_tag(self, ch: str) -> None:
        if self._tag_in_quote:
            # Inside a quoted attribute value: only the matching closing quote
            # ends this sub-state; everything else (including '<') is literal.
            self._tag_buf += ch
            if ch == self._tag_quote_char:
                self._tag_in_quote = False
        elif ch == ">":
            self._tag_buf += ch
            self._dispatch_tag(self._tag_buf)
            self._tag_buf = ""
            self._in_tag = False
        elif ch == "<":
            # A new '<' arrived before the current tag was closed.
            # The buffered content is not a valid tag; absorb it as
            # contextual text/noise and restart tag accumulation.
            self._flush_stale_tag_buf()
            self._tag_buf = "<"
            # _in_tag stays True; _tag_in_quote is already False
        elif ch in ('"', "'"):
            self._tag_buf += ch
            self._tag_in_quote = True
            self._tag_quote_char = ch
        else:
            self._tag_buf += ch
            # Early rejection: if this prefix cannot possibly match any
            # structural tag, flush the buffer now instead of waiting for ">".
            # Common cases: "<--", "<!", "< ", "<br>", "<a ", etc.
            if not self._tag_prefix_is_viable():
                self._flush_stale_tag_buf()
                self._in_tag = False

    def _feed_param_value(self, ch: str) -> None:
        if ch == "<":
            self._tag_buf = "<"
            self._in_tag = True
        else:
            self._param_buf += ch

    def _feed_param_close_pending(self, ch: str) -> None:
        """Handle a character while waiting for the lookahead after </parameter>."""
        if ch == "<":
            self._tag_buf = "<"
            self._in_tag = True
        else:
            # Buffer for potential rollback if the lookahead fails.
            self._pending_close_ws += ch

    def _feed_tool_call_pending(self, ch: str) -> None:
        """Handle a character while waiting for the lookahead after <tool_call>."""
        if ch == "<":
            self._tag_buf = "<"
            self._in_tag = True
        elif ch in " \t\n\r":
            # Only whitespace is permitted between <tool_call> and the next
            # structural tag.  Buffer it so we can restore it on lookahead
            # failure.
            self._pending_tool_call_ws += ch
        else:
            # Non-whitespace before any structural tag: the <tool_call> was
            # plain text content (e.g. a documentation example).  Roll back
            # immediately — do not wait for a potential <function=…> later.
            self._text_buf += "<tool_call>" + self._pending_tool_call_ws + ch
            self._pending_tool_call_ws = ""
            self._state = _State.TEXT

    def _feed_context(self, ch: str) -> None:
        """Handle a character that is outside a tag and outside a param value."""
        if ch == "<":
            self._tag_buf = "<"
            self._in_tag = True
        elif self._state == _State.TEXT:
            self._text_buf += ch
        # In TOOL_CALL / FUNCTION states whitespace between structural tags is
        # irrelevant; discard it silently.

    # ------------------------------------------------------------------
    # Tag dispatch
    # ------------------------------------------------------------------

    def _dispatch_tag(self, tag: str) -> None:
        stripped = tag.strip()
        if self._state == _State.PARAM_VALUE:
            self._dispatch_tag_in_param(stripped, tag)
        elif self._state == _State.PARAM_CLOSE_PENDING:
            self._dispatch_tag_after_param_close(stripped, tag)
        elif self._state == _State.TOOL_CALL_PENDING:
            self._dispatch_tag_after_tool_call_pending(stripped, tag)
        else:
            self._dispatch_structural_tag(stripped, tag)

    def _dispatch_tag_in_param(self, stripped: str, raw: str) -> None:
        """
        A complete tag was read while inside a parameter value.
        Only structural *closing* tags end the value; anything else is literal
        content (e.g. HTML tags, XML fragments in JSON strings, etc.).

        ``</parameter>`` uses one-tag lookahead: it is only a genuine delimiter
        when followed by another ``<parameter=…>`` or by ``</function>`` /
        ``</tool_call>``.  The decision is deferred to
        ``_dispatch_tag_after_param_close``.
        """
        if stripped == "</parameter>":
            # Defer: wait to see what the next tag is before committing.
            self._state = _State.PARAM_CLOSE_PENDING
            self._pending_close_ws = ""
            return
        elif stripped == "</function>":
            self._end_parameter()
            self._end_function()
            self._state = _State.TOOL_CALL
        elif stripped == "</tool_call>":
            self._end_parameter()
            self._end_function()
            self._end_tool_call()
            self._state = _State.TEXT
        else:
            self._param_buf += raw

    def _dispatch_tag_after_param_close(self, stripped: str, raw: str) -> None:
        """
        Lookahead handler: called when we see a complete tag after a tentative
        ``</parameter>``.

        * ``<parameter=…>``  → the ``</parameter>`` was a genuine delimiter;
          end the current parameter and open the next one.
        * ``</function>`` / ``</tool_call>`` → also genuine; end parameter and
          continue the normal structural sequence.
        * Anything else → the ``</parameter>`` was literal content inside the
          value; restore it (plus any buffered inter-tag whitespace) and stay
          in PARAM_VALUE.
        """
        pname = self._match_parameter_open(stripped)
        if pname is not None:
            self._end_parameter()
            self._pending_close_ws = ""
            self._start_parameter(pname)
            self._state = _State.PARAM_VALUE
            return

        if stripped == "</function>":
            self._end_parameter()
            self._pending_close_ws = ""
            self._end_function()
            self._state = _State.TOOL_CALL
            return

        if stripped == "</tool_call>":
            self._end_parameter()
            self._pending_close_ws = ""
            self._end_function()
            self._end_tool_call()
            self._state = _State.TEXT
            return

        # Lookahead failed: the </parameter> was content, not a delimiter.
        self._param_buf += "</parameter>" + self._pending_close_ws + raw
        self._pending_close_ws = ""
        self._state = _State.PARAM_VALUE

    def _dispatch_tag_after_tool_call_pending(self, stripped: str, raw: str) -> None:
        """
        Lookahead handler: called when we see a complete tag after a tentative
        ``<tool_call>``.

        * ``<function=…>`` → the ``<tool_call>`` was a genuine opener; flush
          any preceding plain text, start the tool call and the function.
        * Anything else → the ``<tool_call>`` was literal content (e.g. the
          model mentioned it in reasoning text); restore it together with the
          buffered inter-tag chars as plain text and re-dispatch the new tag
          from TEXT state.
        """
        fname = self._match_function_open(stripped)
        if fname is not None:
            self._flush_text_as_content()
            if self._call_id is not None:
                self._end_tool_call()
            self._pending_tool_call_ws = ""
            self._start_tool_call()
            self._start_function(fname)
            self._state = _State.FUNCTION
            return

        # Lookahead failed: the <tool_call> was content, not a tool call opener.
        self._text_buf += "<tool_call>" + self._pending_tool_call_ws
        self._pending_tool_call_ws = ""
        self._state = _State.TEXT
        self._dispatch_structural_tag(stripped, raw)

    def _dispatch_structural_tag(self, stripped: str, raw: str) -> None:
        """Dispatch a tag encountered between structural elements."""

        if stripped == "<tool_call>":
            if self._state == _State.TEXT:
                # Lookahead: only commit once we see <function=…> next.
                # Reasoning text often mentions <tool_call> without following
                # it with a function tag; defer the decision.
                self._pending_tool_call_ws = ""
                self._state = _State.TOOL_CALL_PENDING
            else:
                # Already inside a structured context (TOOL_CALL or FUNCTION):
                # a second <tool_call> is a back-to-back call; close the
                # previous one (if open) and start the new one immediately.
                self._flush_text_as_content()
                if self._call_id is not None:
                    self._end_tool_call()
                self._start_tool_call()
                self._state = _State.TOOL_CALL
            return

        if stripped == "</tool_call>":
            if self._call_id is None and self._state == _State.TEXT:
                # Stray </tool_call> with no open call: treat as literal text
                # (e.g. rolled back from TOOL_CALL_PENDING lookahead failure).
                self._text_buf += raw
            else:
                self._end_tool_call()
                self._state = _State.TEXT
            return

        fname = self._match_function_open(stripped)
        if fname is not None:
            if self._call_id is None:
                # <function=…> outside any <tool_call> context: plain text.
                self._text_buf += raw
            else:
                self._start_function(fname)
                self._state = _State.FUNCTION
            return

        if stripped == "</function>":
            self._end_function()
            self._state = _State.TOOL_CALL
            return

        pname = self._match_parameter_open(stripped)
        if pname is not None:
            if self._call_id is None:
                # <parameter=…> outside any <tool_call> context: plain text.
                self._text_buf += raw
            else:
                self._start_parameter(pname)
                self._state = _State.PARAM_VALUE
            return

        if stripped == "</parameter>":
            self._end_parameter()
            self._state = _State.FUNCTION
            return

        # Unknown tag: treat as plain text when in TEXT state, ignore otherwise.
        if self._state == _State.TEXT:
            self._text_buf += raw

    # ------------------------------------------------------------------
    # Structural element handlers
    # ------------------------------------------------------------------

    def _start_tool_call(self) -> None:
        self._call_id = make_tool_call_id()
        self.tool_call_index += 1
        self._params_emitted = []
        self._function_ever_started = False

    def _start_function(self, fname: str) -> None:
        self._function_name = fname
        self._function_ever_started = True
        self._params_emitted = []
        self._emit(
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.tool_call_index - 1,
                        id=self._call_id,
                        type="function",
                        function=DeltaFunctionCall(name=fname, arguments=""),
                    )
                ]
            )
        )

    def _start_parameter(self, pname: str) -> None:
        self._param_name = pname
        self._param_buf = ""
        # Emit the JSON key fragment that precedes the value.
        if not self._params_emitted:
            key_frag = f'{{"{pname}": '
        else:
            key_frag = f', "{pname}": '
        self._emit(
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.tool_call_index - 1,
                        id=self._call_id,
                        type="function",
                        function=DeltaFunctionCall(name=None, arguments=key_frag),
                    )
                ]
            )
        )

    def _end_parameter(self) -> None:
        if self._param_name is None:
            return
        raw = self._param_buf
        # Strip the single leading/trailing newline that the template format
        # places around parameter values.
        if raw.startswith("\n"):
            raw = raw[1:]
        if raw.endswith("\n"):
            raw = raw[:-1]
        param_type = self._get_param_type(self._param_name)
        json_value = self._raw_to_json(raw, param_type)
        self._emit(
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.tool_call_index - 1,
                        id=self._call_id,
                        type="function",
                        function=DeltaFunctionCall(name=None, arguments=json_value),
                    )
                ]
            )
        )
        self._params_emitted.append(self._param_name)
        self._param_name = None
        self._param_buf = ""

    def _end_function(self) -> None:
        if self._function_name is None:
            return
        args = "}" if self._params_emitted else "{}"
        self._emit(
            DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.tool_call_index - 1,
                        id=self._call_id,
                        type="function",
                        function=DeltaFunctionCall(name=None, arguments=args),
                    )
                ]
            )
        )
        self._function_name = None

    def _end_tool_call(self) -> None:
        # Ensure any open param/function is cleanly closed first.
        if self._param_name is not None:
            self._end_parameter()
        if self._function_name is not None:
            self._end_function()
        # Only emit the terminator delta if a function was actually started.
        # Skipping it for empty/aborted calls prevents name=None / args=''
        # ghost tool-call entries from reaching the client.
        if self._function_ever_started:
            self._emit(
                DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self._call_id,
                            type="function",
                            function=DeltaFunctionCall(name=None, arguments=""),
                        )
                    ]
                )
            )
        self._call_id = None
        self._function_name = None
        self._function_ever_started = False
        self._param_name = None
        self._param_buf = ""
        self._params_emitted = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flush_text_as_content(self) -> None:
        if self._text_buf:
            self._emit(DeltaMessage(content=self._text_buf))
            self._text_buf = ""

    def _flush_stale_tag_buf(self) -> None:
        """
        A second '<' arrived before the current tag was closed (or the tag
        could not be parsed).  The buffered partial-tag text is not a valid
        structural tag; absorb it into the appropriate accumulator.
        """
        if self._state == _State.PARAM_VALUE:
            self._param_buf += self._tag_buf
        elif self._state == _State.PARAM_CLOSE_PENDING:
            # The tentative </parameter> was content: restore it along with any
            # inter-tag whitespace and the stale tag fragment, then stay in
            # PARAM_VALUE so normal accumulation continues.
            self._param_buf += "</parameter>" + self._pending_close_ws + self._tag_buf
            self._pending_close_ws = ""
            self._state = _State.PARAM_VALUE
        elif self._state == _State.TOOL_CALL_PENDING:
            # The tentative <tool_call> was content: restore it along with any
            # inter-tag whitespace and the stale tag fragment, then return to
            # TEXT so normal accumulation continues.
            self._text_buf += "<tool_call>" + self._pending_tool_call_ws + self._tag_buf
            self._pending_tool_call_ws = ""
            self._state = _State.TEXT
        elif self._state == _State.TEXT:
            self._text_buf += self._tag_buf
        # In TOOL_CALL / FUNCTION: discard (structural inter-tag noise).
        self._tag_buf = ""
        self._tag_in_quote = False
        self._tag_quote_char = ""

    def _match_function_open(self, stripped: str) -> Optional[str]:
        m = self._RE_FUNCTION_OPEN.match(stripped)
        if m:
            return (m.group(1) or m.group(2) or "").strip()
        return None

    def _match_parameter_open(self, stripped: str) -> Optional[str]:
        m = self._RE_PARAMETER_OPEN.match(stripped)
        if m:
            return (m.group(1) or m.group(2) or "").strip()
        return None

    def _get_param_type(self, param_name: str) -> str:
        if not self.tools or not self._function_name:
            return "string"
        props = find_tool_properties(self.tools, self._function_name)
        if param_name in props and isinstance(props[param_name], dict):
            return self._canonical_type(str(props[param_name].get("type", "string")))
        return "string"

    @staticmethod
    def _canonical_type(t: str) -> str:
        t = t.strip().lower()
        if t in ("string", "str", "text", "varchar", "char", "enum"):
            return "string"
        if any(t.startswith(p) for p in ("int", "uint", "long", "short", "unsigned")):
            return "integer"
        if any(t.startswith(p) for p in ("num", "float")):
            return "number"
        if t in ("boolean", "bool", "binary"):
            return "boolean"
        if t == "object" or t.startswith("dict"):
            return "object"
        if t in ("array", "arr", "sequence") or t.startswith("list"):
            return "array"
        return "string"

    def _raw_to_json(self, raw: str, param_type: str) -> str:
        """Convert a raw (unescaped) parameter value to a JSON value fragment."""
        if raw.lower() == "null":
            return "null"

        if param_type == "string":
            return json.dumps(raw, ensure_ascii=False)

        if param_type == "integer":
            try:
                return str(int(float(raw)))
            except (ValueError, TypeError):
                pass
            return json.dumps(raw, ensure_ascii=False)

        if param_type == "number":
            try:
                v = float(raw)
                return str(int(v)) if v == int(v) else repr(v)
            except (ValueError, TypeError):
                pass
            return json.dumps(raw, ensure_ascii=False)

        if param_type == "boolean":
            return "true" if raw.strip().lower() == "true" else "false"

        if param_type in ("object", "array"):
            # Try strict JSON first, then ast.literal_eval for Python-style
            # containers that models sometimes emit ({'key': 'val'} etc.).
            try:
                return json.dumps(json.loads(raw), ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                return json.dumps(ast.literal_eval(raw), ensure_ascii=False)
            except Exception:
                pass
            return json.dumps(raw, ensure_ascii=False)

        return json.dumps(raw, ensure_ascii=False)

    def _emit(self, delta: DeltaMessage) -> None:
        self.deltas.append(delta)

    @staticmethod
    def _merge_deltas(deltas: list[DeltaMessage]) -> DeltaMessage:
        """
        Merge a list of deltas into a single DeltaMessage.

        Creates new objects rather than mutating the inputs, so it is safe
        to call multiple times on overlapping delta slices.

        The streaming API contract requires that ``id`` and ``type`` are only
        present on the *first* delta for each tool call (the one that carries
        the function name).  Subsequent argument-only deltas must have
        ``id=None`` so the client knows they extend an existing call rather
        than open a new one.  This method enforces that invariant regardless
        of how many raw deltas are being merged.
        """
        if not deltas:
            return DeltaMessage(content=None)

        merged_content = ""
        # Preserve insertion order while merging per call-id.
        call_order: list[str] = []
        call_map: dict[str, dict] = {}

        for d in deltas:
            if d.content:
                merged_content += d.content
            if d.tool_calls:
                for tc in d.tool_calls:
                    cid = tc.id or ""
                    if cid not in call_map:
                        call_map[cid] = {
                            "index": tc.index,
                            "type": tc.type or "function",
                            "name": None,
                            "args": [],
                        }
                        call_order.append(cid)
                    entry = call_map[cid]
                    if tc.type:
                        entry["type"] = tc.type
                    if tc.index is not None:
                        entry["index"] = tc.index
                    if tc.function:
                        if tc.function.name:
                            entry["name"] = tc.function.name
                        if tc.function.arguments is not None:
                            entry["args"].append(tc.function.arguments)

        merged_calls = [
            DeltaToolCall(
                index=call_map[cid]["index"],
                # id and type are only emitted on the first delta for a tool
                # call (the one carrying the function name).  Subsequent
                # argument-only deltas must have id=None so the streaming
                # reconstructor knows they extend an existing call rather than
                # opening a new one.
                id=cid if call_map[cid]["name"] else None,
                type=call_map[cid]["type"] if call_map[cid]["name"] else None,
                function=DeltaFunctionCall(
                    name=call_map[cid]["name"],
                    arguments="".join(call_map[cid]["args"]),
                ),
            )
            for cid in call_order
        ]

        if merged_calls:
            return DeltaMessage(
                content=merged_content or None,
                tool_calls=merged_calls,
            )
        return DeltaMessage(content=merged_content or None)


class Qwen3XMLToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.parser = StreamingXMLToolCallParser()

        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        logger.info(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        self.parser.reset_streaming_state()
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self.parser.set_tools(self.tools)

        # Feed the complete output, then finalize to close any open elements.
        self.parser.parse_single_streaming_chunks(model_output)
        self.parser.finalize()

        # Collect the fully merged result.
        full = self.parser.collect_all()

        if not full.tool_calls:
            return ExtractedToolCallInformation(
                tool_calls=[],
                tools_called=False,
                content=full.content,
            )

        tool_calls: list[ToolCall] = []
        for tc in full.tool_calls:
            if tc.function and tc.function.name:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function=FunctionCall(
                            name=tc.function.name,
                            arguments=tc.function.arguments or "",
                        ),
                    )
                )
                idx = (
                    tc.index
                    if tc.index is not None
                    else len(self.prev_tool_call_arr) - 1
                )
                while len(self.prev_tool_call_arr) <= idx:
                    self.prev_tool_call_arr.append({"name": "", "arguments": ""})
                while len(self.streamed_args_for_tool) <= idx:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[idx]["name"] = tc.function.name
                self.prev_tool_call_arr[idx]["arguments"] = tc.function.arguments or ""
                if tc.function.arguments:
                    self.streamed_args_for_tool[idx] = tc.function.arguments

        return ExtractedToolCallInformation(
            tool_calls=tool_calls,
            tools_called=len(tool_calls) > 0,
            content=full.content,
        )

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
        if not previous_text:
            self.parser.reset_streaming_state()
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = []
            self.parser.set_tools(self.tools)

        # Empty delta text but real tokens: detect stream-end conditions.
        if not delta_text and delta_token_ids:
            open_calls = current_text.count(
                self.parser.tool_call_start_token
            ) - current_text.count(self.parser.tool_call_end_token)
            if (
                open_calls == 0
                and self.parser.tool_call_index > 0
                or not self.parser.tool_call_index
                and current_text
            ):
                return DeltaMessage(content="")
            return None

        delta = self.parser.parse_single_streaming_chunks(delta_text)

        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function:
                    idx = (
                        tc.index
                        if tc.index is not None
                        else len(self.prev_tool_call_arr) - 1
                    )
                    while len(self.prev_tool_call_arr) <= idx:
                        self.prev_tool_call_arr.append({"name": "", "arguments": ""})
                    while len(self.streamed_args_for_tool) <= idx:
                        self.streamed_args_for_tool.append("")
                    if tc.function.name:
                        self.prev_tool_call_arr[idx]["name"] = tc.function.name
                    if tc.function.arguments is not None:
                        self.prev_tool_call_arr[idx]["arguments"] += (
                            tc.function.arguments
                        )
                        self.streamed_args_for_tool[idx] += tc.function.arguments

        if delta.content is None and not delta.tool_calls and delta.reasoning is None:
            return None
        return delta
