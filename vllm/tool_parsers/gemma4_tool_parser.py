# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tool call parser for Google Gemma4 models.

Gemma4 uses a custom serialization format (not JSON) for tool calls::

    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

Strings are delimited by ``<|"|>`` (token 52), keys are unquoted, and
multiple tool calls are concatenated without separators.

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

    # Null
    if value_str.lower() in ("null", "none", "nil"):
        return None

    # Number (int or float)
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    return value_str


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

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
            if not partial:
                result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
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
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
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
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 args parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            if partial:
                raw_val = args_str[val_start:i].strip()
                if raw_val.endswith("."):
                    # Trailing dot means decimal digits may still arrive
                    # (e.g. "108." may become "108.2"). Parsing now would
                    # yield float("108.") == 108.0, whose json repr "108.0"
                    # corrupts the streaming diff when the true digit lands.
                    break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
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
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 array parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            if partial:
                raw_val = arr_str[val_start:i].strip()
                if raw_val.endswith("."):
                    break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Google Gemma4 models.

    Handles the Gemma4 function call format::

        <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>

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

        # Regex for non-streaming: extract complete tool calls.
        # Supports function names with letters, digits, underscores,
        # hyphens, and dots (e.g. "get-weather", "module.func").
        self.tool_call_regex = re.compile(
            r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>",
            re.DOTALL,
        )

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

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Don't skip special tokens — <|tool_call> etc. are needed for
            # the parser to detect tool calls. Apply to BOTH
            # ChatCompletionRequest and ResponsesRequest (the previous
            # isinstance(ChatCompletionRequest) guard caused tool-call
            # delimiters to be stripped on /v1/responses, leaking raw
            # `call:fn{...}` text via output_text.delta).
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

        # Check if combined ends with a complete special token
        if combined.endswith(TOOL_CALL_START) or combined.endswith(TOOL_CALL_END):
            self.buffered_delta_text = ""
            return combined

        # Check if combined ends with a partial prefix of a special token
        for tag in [TOOL_CALL_START, TOOL_CALL_END]:
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
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            matches = self.tool_call_regex.findall(model_output)
            if not matches:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls: list[ToolCall] = []
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

            # Content = text before first tool call (if any)
            content_end = model_output.find(self.tool_call_start_token)
            content = model_output[:content_end].strip() if content_end > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
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
        # Buffer delta text to handle multi-token special sequences
        delta_text = self._buffer_delta_text(delta_text)
        # Keep current_text from the upstream stream state. The buffered delta
        # is only for emission, and must not be stitched back into the
        # accumulated model text or normal content like "<div>" can be
        # duplicated into "<<div>" when a tool call just ended.

        # If no tool call token seen yet, emit as content
        if self.tool_call_start_token not in current_text:
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        try:
            return self._extract_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
            )
        except Exception:
            logger.exception("Error in Gemma4 streaming tool call extraction")
            return None

    def _extract_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Streaming parser using full-match enumeration (Hermes-style).

        Instead of relying on tag counting for phase detection, this method:
        1. Uses regex findall to enumerate all **complete** tool calls in
           current_text.
        2. Compares against prev_tool_call_arr length to discover new ones.
        3. Handles partial (incomplete) tool calls at the end separately.

        This correctly handles the case where multiple tool calls appear in a
        single delta (e.g.,
        ``<|tool_call>call:f1{...}<tool_call|><|tool_call>call:f2{...}<tool_call|>``).

        Format: ``<|tool_call>call:name{args}<tool_call|>``
        """
        all_matches = self.tool_call_regex.findall(current_text)
        curr_start_count = current_text.count(self.tool_call_start_token)
        curr_end_count = current_text.count(self.tool_call_end_token)
        prev_start_count = previous_text.count(self.tool_call_start_token)

        tool_call_deltas: list[DeltaToolCall] = []

        # Step 1: Process all complete tool calls
        for i, (func_name, args_str) in enumerate(all_matches):
            deltas = self._process_complete_call(i, func_name, args_str)
            tool_call_deltas.extend(deltas)

        # Step 2: Handle partial (incomplete) tool call
        if curr_start_count > curr_end_count:
            partial_idx = len(all_matches)
            deltas = self._process_partial_call(current_text, partial_idx)
            tool_call_deltas.extend(deltas)

        # Step 3: Handle content
        content_delta = self._compute_content_delta(
            current_text, previous_text, curr_start_count, prev_start_count, delta_text
        )

        # Step 4: Return result
        if content_delta or tool_call_deltas:
            return DeltaMessage(
                content=content_delta,
                tool_calls=tool_call_deltas or [],
            )
        return None

    def _ensure_state_sized(self, index: int) -> None:
        """Ensure state arrays have enough room for the given index."""
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})
            self.streamed_args_for_tool.append("")

    def _process_complete_call(
        self, index: int, func_name: str, args_str: str
    ) -> list[DeltaToolCall]:
        """Process a complete tool call: send name (once) + args diff."""
        self._ensure_state_sized(index)
        deltas: list[DeltaToolCall] = []

        # Send function name if not yet sent
        if "name" not in self.prev_tool_call_arr[index]:
            self.prev_tool_call_arr[index]["name"] = func_name
            deltas.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(name=func_name),
                )
            )

        # Parse arguments and diff against previously streamed
        final_args = _parse_gemma4_args(args_str)
        final_args_json = json.dumps(final_args, ensure_ascii=False)
        prev_streamed = self.streamed_args_for_tool[index]

        if len(final_args_json) > len(prev_streamed):
            diff = final_args_json[len(prev_streamed):]
            self.streamed_args_for_tool[index] = final_args_json
            self.prev_tool_call_arr[index]["arguments"] = final_args
            deltas.append(
                DeltaToolCall(
                    index=index,
                    function=DeltaFunctionCall(arguments=diff),
                )
            )

        # Track the last known tool call
        self.current_tool_id = index
        self.current_tool_name_sent = True

        return deltas

    def _process_partial_call(
        self, current_text: str, partial_idx: int
    ) -> list[DeltaToolCall]:
        """Process an incomplete/partial tool call at the end of text."""
        self._ensure_state_sized(partial_idx)
        deltas: list[DeltaToolCall] = []

        func_name, args_part = self._extract_partial_call(current_text)

        if func_name is not None:
            # Send function name if not yet sent
            if "name" not in self.prev_tool_call_arr[partial_idx]:
                self.prev_tool_call_arr[partial_idx]["name"] = func_name
                deltas.append(
                    DeltaToolCall(
                        index=partial_idx,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(name=func_name),
                    )
                )
                self.current_tool_id = partial_idx
                self.current_tool_name_sent = True

            # Stream argument diff
            if self.current_tool_name_sent and args_part:
                partial_delta = self._emit_argument_diff_for_index(
                    partial_idx, args_part
                )
                if partial_delta:
                    deltas.append(partial_delta)

        return deltas

    def _compute_content_delta(
        self,
        current_text: str,
        previous_text: str,
        curr_start_count: int,
        prev_start_count: int,
        delta_text: str,
    ) -> str | None:
        """Extract plain text content by removing tool call patterns."""
        # Case A: No tool call tokens at all — treat as pure content
        if curr_start_count == 0 and prev_start_count == 0:
            return delta_text if delta_text else None

        # Remove all complete tool call patterns
        content_current = self.tool_call_regex.sub("", current_text)
        content_previous = self.tool_call_regex.sub("", previous_text)

        # Strip any trailing partial tool call
        content_current = self._strip_trailing_partial(content_current)
        content_previous = self._strip_trailing_partial(content_previous)

        # Compute the diff
        if content_current.startswith(content_previous):
            result = content_current[len(content_previous):]
        elif content_previous.startswith(content_current):
            result = content_current
        else:
            result = content_current

        return result if result else None

    def _strip_trailing_partial(self, text: str) -> str:
        """Remove any trailing partial tool call (from <|tool_call> to end)."""
        idx = text.rfind(self.tool_call_start_token)
        if idx != -1:
            return text[:idx]
        return text

    def _emit_argument_diff_for_index(
        self, index: int, raw_args_str: str
    ) -> DeltaToolCall | None:
        """Like _emit_argument_diff but accepts an explicit index.

        Parse raw Gemma4 arguments for a specific tool call, convert to JSON,
        diff against previously streamed args, and emit only the new chars.
        """
        try:
            current_args = _parse_gemma4_args(raw_args_str, partial=True)
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

        prev_streamed = self.streamed_args_for_tool[index]

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
                # tracking to the common prefix.
                self.streamed_args_for_tool[index] = prefix
                return None

            # Stream the new stable portion
            diff = safe_json[sent_len:]
        else:
            # First emission
            diff = safe_json

        if diff:
            self.streamed_args_for_tool[index] = safe_json
            self.prev_tool_call_arr[index]["arguments"] = current_args

            return DeltaToolCall(
                index=index,
                id=None,
                function=DeltaFunctionCall(arguments=diff),
            )

        return None

    def _extract_partial_call(self, current_text: str) -> tuple[str | None, str]:
        """Extract function name and raw argument string from partial text.

        Returns (func_name, raw_args_str) or (None, "") if not parseable yet.
        """
        # Get the text after the last <|tool_call> token
        last_start = current_text.rfind(self.tool_call_start_token)
        if last_start == -1:
            return None, ""

        partial_call = current_text[last_start + len(self.tool_call_start_token) :]

        # Strip end token if present
        if self.tool_call_end_token in partial_call:
            partial_call = partial_call.split(self.tool_call_end_token)[0]

        # Expect "call:name{args...}" or "call:name{args...}"
        if not partial_call.startswith("call:"):
            return None, ""

        func_part = partial_call[5:]  # skip "call:"

        if "{" not in func_part:
            # Still accumulating function name, not ready yet
            return None, ""

        func_name, _, args_part = func_part.partition("{")
        func_name = func_name.strip()

        # Strip trailing '}' if present (Gemma4 structural brace)
        if args_part.endswith("}"):
            args_part = args_part[:-1]

        return func_name, args_part

