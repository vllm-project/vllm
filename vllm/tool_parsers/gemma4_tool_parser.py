# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tool call parser for Google Gemma4 models.

Handles TWO tool call formats emitted by Gemma4:

FORMAT 1 — Native token format (primary):
    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

FORMAT 2 — Fallback XML-style format (seen when the model regresses):
    <tool_call>hangup_call{}</tool_call>
    <tool_call>func_name{key:<|"|>value<|"|>}</tool_call>

The fallback format differs from the native format in three ways:
  - Uses plain ``<tool_call>`` / ``</tool_call>`` tags instead of
    ``<|tool_call>`` / ``<tool_call|>``
  - Does NOT include the ``call:`` prefix before the function name
  - Is otherwise structurally identical (same key:value / <|"|> encoding)

Both formats share the same Gemma4 argument encoding, so
``_parse_gemma4_args()`` is reused for both.


Used when ``--enable-auto-tool-choice --tool-call-parser gemma4`` are set.

For offline inference tool call parsing (direct ``tokenizer.decode()`` output),
see ``vllm.tool_parsers.gemma4_utils.parse_tool_calls``.
"""

import json
from collections.abc import Sequence

import regex as re

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
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import find_common_prefix

logger = init_logger(__name__)

# Gemma4 special tokens for tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'


# ---------------------------------------------------------------------------
# Fallback format constants
# ---------------------------------------------------------------------------
FALLBACK_TOOL_CALL_START = "<tool_call>"
FALLBACK_TOOL_CALL_END = "</tool_call>"

# Regex for native format (complete match, non-streaming)
#   <|tool_call>call:func_name{...}<tool_call|>
_NATIVE_REGEX = re.compile(
    r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)

# Regex for fallback format (complete match, non-streaming)
#   <tool_call>func_name{...}</tool_call>
#   Works for empty args too:  <tool_call>hangup_call{}</tool_call>
_FALLBACK_REGEX = re.compile(
    r"<tool_call>([\w\-\.]+)\{(.*?)\}</tool_call>",
    re.DOTALL,
)
# ---------------------------------------------------------------------------
# Gemma4 argument parser (used by both streaming and non-streaming paths)
# ---------------------------------------------------------------------------


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Number (int or float)
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    return value_str


def _parse_gemma4_args(args_str: str) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        # Parse value
        if i >= n:
            result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    # Skip over string contents to avoid counting { inside strings
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            arr_content = args_str[arr_start : i - 1]
            result[key] = _parse_gemma4_array(arr_content)

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str) -> list:
    """Parse a Gemma4 array content string into a Python list."""
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # String element
        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items



# ---------------------------------------------------------------------------
# Helper: detect which format is present
# ---------------------------------------------------------------------------

def _detect_format(text: str) -> str:
    """Return 'native', 'fallback', 'both', or 'none'."""
    has_native = TOOL_CALL_START in text
    has_fallback = FALLBACK_TOOL_CALL_START in text and FALLBACK_TOOL_CALL_END in text
    if has_native and has_fallback:
        return "both"
    if has_native:
        return "native"
    if has_fallback:
        return "fallback"
    return "none"


def _build_tool_calls(matches: list[tuple[str, str]]) -> list[ToolCall]:
    """Convert (func_name, args_str) match pairs into ToolCall objects."""
    tool_calls = []
    for func_name, args_str in matches:
        arguments = _parse_gemma4_args(args_str)
        tool_calls.append(
            ToolCall(
                type="function",
                function=FunctionCall(
                    name=func_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        )
    return tool_calls


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Google Gemma4 models.

    Handles the Gemma4 function call format::

        <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>

     **Fallback XML-style** (seen when the model regresses or uses a
    non-native chat template)::

        <tool_call>hangup_call{}</tool_call>
        <tool_call>func_name{key:<|"|>value<|"|>}</tool_call>

    Used when ``--enable-auto-tool-choice --tool-call-parser gemma4``
    are set.

    Streaming strategy: **accumulate-then-parse-then-diff**

    Instead of trying to convert Gemma4's custom format to JSON
    token-by-token (which fails because Gemma4 uses bare keys, custom
    delimiters, and structural braces that differ from JSON), this parser:

    1. Accumulates the raw Gemma4 argument string during streaming
    2. Parses it with ``_parse_gemma4_args()`` into a Python dict
    3. Converts to JSON with ``json.dumps()``
    4. Diffs against the previously-streamed JSON string
    5. Emits only the new JSON fragment as the delta

    This follows the same pattern used by FunctionGemma, Hermes, and Llama
    tool parsers.
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Token strings
        self.tool_call_start_token = TOOL_CALL_START
        self.tool_call_end_token = TOOL_CALL_END

        # Token IDs
        self.tool_call_start_token_id = self.vocab.get(TOOL_CALL_START)
        self.tool_call_end_token_id = self.vocab.get(TOOL_CALL_END)

        if self.tool_call_start_token_id is None:
            raise RuntimeError(
                "Gemma4 ToolParser could not locate the tool call start "
                f"token '{TOOL_CALL_START}' in the tokenizer!"
            )

         # Native and fallback regexes (non-streaming, complete match)
        self.tool_call_regex = _NATIVE_REGEX
        self.fallback_tool_call_regex = _FALLBACK_REGEX

        # Streaming state — reset per-request via _reset_streaming_state()
        self._reset_streaming_state()

        # Delta buffer for handling multi-token special sequences
        self.buffered_delta_text = ""

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new request."""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        # Track which format the current stream is using
        self._streaming_format: str = "none"

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if (
            isinstance(request, ChatCompletionRequest)
            and request.tools
            and request.tool_choice != "none"
        ):
            # Don't skip special tokens — <|tool_call> etc. are needed
            request.skip_special_tokens = False
        return request

    # ------------------------------------------------------------------
    # Delta buffering for multi-token special sequences
    # ------------------------------------------------------------------

    def _buffer_delta_text(self, delta_text: str) -> str:
        """Buffer incoming delta text to handle multi-token special sequences.

        Accumulates partial tokens that could be the start of
        ``<|tool_call>`` or ``<tool_call|>`` and only flushes them
        when the complete sequence is recognized or the sequence breaks.

        This prevents partial special tokens (e.g., ``<|tool``) from being
        emitted prematurely as content text.
        """
        combined = self.buffered_delta_text + delta_text

        all_tags = [
            TOOL_CALL_START, TOOL_CALL_END,
            FALLBACK_TOOL_CALL_START, FALLBACK_TOOL_CALL_END,
        ]

        # Check if combined ends with a complete special token
        if any(combined.endswith(tag) for tag in all_tags):
            self.buffered_delta_text = ""
            return combined

        # Check if combined ends with a partial prefix of a special token
        for tag in all_tags:
            for i in range(1, len(tag)):
                if combined.endswith(tag[:i]):
                    self.buffered_delta_text = combined[-i:]
                    return combined[:-i]

        # No partial match — flush everything
        self.buffered_delta_text = ""
        return combined

    # ------------------------------------------------------------------
    # Non-streaming extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete model output string.

        Tries the native format first; falls back to the XML-style format
        if no native tool calls are found.
        """
        fmt = _detect_format(model_output)

        if fmt == "none":
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
       # --- Native format ---
            if fmt in ("native", "both"):
                matches = self.tool_call_regex.findall(model_output)
                if matches:
                    tool_calls = _build_tool_calls(matches)
                    content_end = model_output.find(TOOL_CALL_START)
                    content = model_output[:content_end].strip() or None
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=tool_calls,
                        content=content,
                    )

            # --- Fallback XML-style format ---
            if fmt in ("fallback", "both"):
                matches = self.fallback_tool_call_regex.findall(model_output)
                if matches:
                    logger.debug(
                        "Gemma4ToolParser: using fallback XML format for %d tool call(s)",
                        len(matches),
                    )
                    tool_calls = _build_tool_calls(matches)
                    content_end = model_output.find(FALLBACK_TOOL_CALL_START)
                    content = model_output[:content_end].strip() or None
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=tool_calls,
                        content=content,
                    )


        except Exception:
            logger.exception("Error extracting tool calls from Gemma4 response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ------------------------------------------------------------------
    # Streaming extraction — accumulate-then-parse-then-diff
    # ------------------------------------------------------------------

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
        # Buffer delta text to handle multi-token tag sequences
        delta_text = self._buffer_delta_text(delta_text)
        # Keep current_text from the upstream stream state. The buffered delta
        # is only for emission, and must not be stitched back into the
        # accumulated model text or normal content like "<div>" can be
        # duplicated into "<<div>" when a tool call just ended.

        # Detect which format is active (latch on first detection)
        if self._streaming_format == "none":
            if TOOL_CALL_START in current_text:
                self._streaming_format = "native"
                logger.debug("Gemma4ToolParser: streaming format = native")
            elif FALLBACK_TOOL_CALL_START in current_text:
                self._streaming_format = "fallback"
                logger.debug("Gemma4ToolParser: streaming format = fallback")


        # If no tool call token seen yet, emit as content
        if self._streaming_format == "none":
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        try:
            if self._streaming_format == "native":
                return self._extract_streaming_native(
                    previous_text=previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                )
            else:  # fallback
                return self._extract_streaming_fallback(
                    previous_text=previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                )
        except Exception:
            logger.exception("Error in Gemma4 streaming tool call extraction")
            return None


    # ------------------------------------------------------------------
    # Native format streaming
    # ------------------------------------------------------------------
    def _extract_streaming_native(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Streaming handler for native <|tool_call>...<tool_call|> format."""
        start_count = current_text.count(TOOL_CALL_START)
        end_count = current_text.count(TOOL_CALL_END)
        prev_start_count = previous_text.count(TOOL_CALL_START)
        prev_end_count = previous_text.count(TOOL_CALL_END)

        # Case 1: Not inside any tool call — emit as content
        if (
            start_count == end_count
            and prev_end_count == end_count
            and TOOL_CALL_END not in delta_text
        ):
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        # Case 2: Starting a new tool call
        if start_count > prev_start_count and start_count > end_count:
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append({})
            logger.debug("Starting new tool call %d", self.current_tool_id)
            # Don't return yet — fall through to try parsing if there's
            # content after <|tool_call> in this same delta
            # (but usually it's just the token itself, so return None)
            if len(delta_text) <= len(TOOL_CALL_START):
                return None

        # Case 3: Tool call just ended
        if end_count > prev_end_count:
            return self._handle_tool_call_end(current_text,self.tool_call_regex, TOOL_CALL_START)

        # Case 4: In the middle of a tool call — parse partial content
        if start_count > end_count:
            return self._handle_tool_call_middle_native(current_text)

        # Default: generate text outside tool calls
        if delta_text:
            text = delta_text.replace(TOOL_CALL_START, "").replace(TOOL_CALL_END, "")
            if text:
                return DeltaMessage(content=text)
        return None

    def _handle_tool_call_middle_native(
        self, current_text: str
    ) -> DeltaMessage | None:
        """Parse partial native-format tool call and emit name/arg deltas."""
        last_start = current_text.rfind(TOOL_CALL_START)
        if last_start == -1:
            return None

        partial_call = current_text[last_start + len(TOOL_CALL_START):]
        if TOOL_CALL_END in partial_call:
            partial_call = partial_call.split(TOOL_CALL_END)[0]

        if not partial_call.startswith("call:"):
            return None

        func_part = partial_call[5:]  # skip "call:"
        if "{" not in func_part:
            return None

        func_name, _, args_part = func_part.partition("{")
        func_name = func_name.strip()
        if args_part.endswith("}"):
            args_part = args_part[:-1]

        return self._emit_name_then_args(func_name, args_part)


    # ------------------------------------------------------------------
    # Fallback format streaming
    # ------------------------------------------------------------------

    def _extract_streaming_fallback(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Streaming handler for fallback <tool_call>...</tool_call> format."""
        start_count = current_text.count(FALLBACK_TOOL_CALL_START)
        end_count = current_text.count(FALLBACK_TOOL_CALL_END)
        prev_start_count = previous_text.count(FALLBACK_TOOL_CALL_START)
        prev_end_count = previous_text.count(FALLBACK_TOOL_CALL_END)

        # Outside any tool call — emit as content
        if (
            start_count == end_count
            and prev_end_count == end_count
            and FALLBACK_TOOL_CALL_END not in delta_text
        ):
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        # New tool call starting
        if start_count > prev_start_count and start_count > end_count:
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append({})
            if len(delta_text) <= len(FALLBACK_TOOL_CALL_START):
                return None

        # Tool call just ended
        if end_count > prev_end_count:
            return self._handle_tool_call_end(
                current_text, self.fallback_tool_call_regex, FALLBACK_TOOL_CALL_START
            )

        # Inside a tool call — parse partial content
        if start_count > end_count:
            return self._handle_tool_call_middle_fallback(current_text)

        if delta_text:
            text = (
                delta_text
                .replace(FALLBACK_TOOL_CALL_START, "")
                .replace(FALLBACK_TOOL_CALL_END, "")
            )
            if text:
                return DeltaMessage(content=text)
        return None

    def _handle_tool_call_middle_fallback(
        self, current_text: str
    ) -> DeltaMessage | None:
        """Parse partial fallback-format tool call and emit name/arg deltas."""
        last_start = current_text.rfind(FALLBACK_TOOL_CALL_START)
        if last_start == -1:
            return None

        partial_call = current_text[last_start + len(FALLBACK_TOOL_CALL_START):]
        if FALLBACK_TOOL_CALL_END in partial_call:
            partial_call = partial_call.split(FALLBACK_TOOL_CALL_END)[0]

        # Fallback format: func_name{args...}  (NO "call:" prefix)
        if "{" not in partial_call:
            return None

        func_name, _, args_part = partial_call.partition("{")
        func_name = func_name.strip()
        if args_part.endswith("}"):
            args_part = args_part[:-1]

        return self._emit_name_then_args(func_name, args_part)

    def _emit_name_then_args(
        self, func_name: str, args_part: str
    ) -> DeltaMessage | None:
        """Emit function name delta (once), then argument deltas."""
        if not func_name:
            return None

        # Step 1: send function name once
        if not self.current_tool_name_sent:
            self.current_tool_name_sent = True
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": func_name,
                "arguments": {},
            }
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=func_name,
                            arguments="",
                        ).model_dump(exclude_none=True),
                    )
                ]
            )

        # Step 2: diff and stream arguments
        if args_part:
            return self._emit_argument_diff(args_part)

        return None


    def _handle_tool_call_end(self,  current_text: str,regex: re.Pattern,start_token: str) -> DeltaMessage | None:
        """Handle streaming when a tool call has just completed.

        Performs a final parse of the complete tool call and flushes
        any remaining un-streamed argument fragments.
        """
        if self.current_tool_id < 0 or self.current_tool_id >= len(
            self.prev_tool_call_arr
        ):
            logger.debug(
                "Tool call end detected but no active tool call (current_tool_id=%d)",
                self.current_tool_id,
            )
            return None

        # Parse the complete tool call using regex for accuracy
        all_matches = self.tool_call_regex.findall(current_text)
        if self.current_tool_id < len(all_matches):
            _, args_str = all_matches[self.current_tool_id]
            final_args = _parse_gemma4_args(args_str)
            final_args_json = json.dumps(final_args, ensure_ascii=False)

            prev_streamed = self.streamed_args_for_tool[self.current_tool_id]
            if len(final_args_json) > len(prev_streamed):
                diff = final_args_json[len(prev_streamed) :]
                self.streamed_args_for_tool[self.current_tool_id] = final_args_json
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = final_args

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                )

        return None

    def _emit_argument_diff(self, raw_args_str: str) -> DeltaMessage | None:
        """Parse raw Gemma4 arguments, convert to JSON, diff, and emit.

        This is the core of the accumulate-then-parse-then-diff strategy:
        1. Parse ``raw_args_str`` with ``_parse_gemma4_args()``
        2. Convert to JSON string with ``json.dumps()``
        3. Withhold trailing closing characters (``"}``) that may move
           as more tokens arrive
        4. Diff against previously streamed JSON and emit only new chars

        **Why withholding is necessary:**

        Gemma4's custom format produces *structurally incomplete* JSON
        during streaming. For example, when ``<|"|>Paris`` arrives
        without a closing delimiter, ``_parse_gemma4_args`` treats it
        as a complete value and produces ``{"location": "Paris"}``. But
        when ``, France<|"|>`` arrives next, the JSON becomes
        ``{"location": "Paris, France"}``. If we had sent the closing
        ``"}`` from the first parse, the concatenated client output
        would be ``{"location": "Paris"}France"}``, which is garbage.

        The solution: **never send trailing closing chars during
        streaming**. They get flushed by ``_handle_tool_call_end()``
        when the ``<tool_call|>`` end marker arrives.

        Args:
            raw_args_str: The raw Gemma4 argument text accumulated so far
                (without the surrounding ``{`` ``}``).

        Returns:
            DeltaMessage with the argument diff, or None if no new content.
        """
        try:
            current_args = _parse_gemma4_args(raw_args_str)
        except Exception:
            logger.debug(
                "Could not parse partial Gemma4 args yet: %s",
                raw_args_str[:100],
            )
            return None

        if not current_args:
            return None

        current_args_json = json.dumps(current_args, ensure_ascii=False)

        # Withhold trailing closing characters that may shift as more
        # tokens arrive. Strip trailing '}', '"', ']' and partial
        # STRING_DELIM fragments ('<', '|', '\\', '>') to get the
        # "safe prefix".
        safe_json = current_args_json
        while safe_json and safe_json[-1] in ("}", '"', "]", "<", "|", "\\", ">"):
            safe_json = safe_json[:-1]

        prev_streamed = self.streamed_args_for_tool[self.current_tool_id]

        if not safe_json or safe_json == prev_streamed:
            return None

        # Use find_common_prefix to handle cases where the value changed
        # structurally (e.g., a string grew).
        if prev_streamed:
            prefix = find_common_prefix(prev_streamed, safe_json)
            sent_len = len(prev_streamed)
            prefix_len = len(prefix)

            if prefix_len < sent_len:
                # Structure changed — we sent too much. Truncate our
                # tracking to the common prefix and wait for the final
                # flush in _handle_tool_call_end.
                self.streamed_args_for_tool[self.current_tool_id] = prefix
                return None

            # Stream the new stable portion
            diff = safe_json[sent_len:]
        else:
            # First emission
            diff = safe_json

        if diff:
            self.streamed_args_for_tool[self.current_tool_id] = safe_json
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = current_args

            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=diff).model_dump(
                            exclude_none=True
                        ),
                    )
                ]
            )

        return None
