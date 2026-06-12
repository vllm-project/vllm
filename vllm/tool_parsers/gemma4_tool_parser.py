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
from vllm.tool_parsers.utils import find_common_prefix, partial_tag_overlap

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

    Streaming strategy: **scan accumulated text, parse, then diff**

    Instead of trying to convert Gemma4's custom format to JSON
    one token at a time (which fails because Gemma4 uses bare keys, custom
    delimiters, and structural braces that differ from JSON), this parser:

    1. Re-scans the accumulated model output to find tool-call regions
    2. Parses each visible region with ``_parse_gemma4_args()``
    3. Converts arguments to JSON with ``json.dumps()``
    4. Diffs against the JSON prefix already streamed for that tool index
    5. Emits only the new function names, argument fragments, and content

    This follows the same shape used by Hermes/Kimi-style parsers and makes
    the parser insensitive to whether speculative decoding delivers one token
    or several delimiter events in a single streaming delta.
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

        # Streaming state — reset per-request via _reset_streaming_state()
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new request."""
        self.current_tool_id = -1
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self._sent_content_idx = 0

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
            parsed_tool_calls = self._extract_tool_call_regions(
                model_output, include_partial=False
            )
            if not parsed_tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls: list[ToolCall] = []
            for func_name, args_str, _ in parsed_tool_calls:
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
    # Streaming extraction — scan accumulated text, parse, then diff
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
        if not previous_text:
            self._reset_streaming_state()

        try:
            return self._extract_streaming(current_text)
        except Exception:
            logger.exception("Error in Gemma4 streaming tool call extraction")
            return None

    def _extract_streaming(self, current_text: str) -> DeltaMessage | None:
        """Stream content and Gemma4 tool-call deltas from accumulated text."""
        content = self._extract_content(current_text)
        tool_calls = self._extract_tool_call_regions(current_text, include_partial=True)
        tool_call_deltas: list[DeltaToolCall] = []

        for index, (func_name, raw_args_str, is_complete) in enumerate(tool_calls):
            self._ensure_tool_state(index)

            if "name" not in self.prev_tool_call_arr[index]:
                self.prev_tool_call_arr[index] = {
                    "name": func_name,
                    "arguments": {},
                }
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=func_name,
                            arguments="",
                        ).model_dump(exclude_none=True),
                    )
                )

            args_delta = self._compute_args_diff(
                index=index,
                raw_args_str=raw_args_str,
                is_complete=is_complete,
            )
            if args_delta:
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        function=DeltaFunctionCall(arguments=args_delta).model_dump(
                            exclude_none=True
                        ),
                    )
                )

        if content or tool_call_deltas:
            return DeltaMessage(content=content, tool_calls=tool_call_deltas)
        return None

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, holding partial start tags."""
        content_parts: list[str] = []
        pos = self._sent_content_idx
        text_len = len(current_text)

        while pos < text_len:
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                overlap = partial_tag_overlap(
                    current_text[pos:], self.tool_call_start_token
                )
                sendable_end = text_len - overlap
                if sendable_end > pos:
                    content_parts.append(current_text[pos:sendable_end])
                    pos = sendable_end
                break

            if start > pos:
                content_parts.append(current_text[pos:start])

            end = current_text.find(
                self.tool_call_end_token, start + len(self.tool_call_start_token)
            )
            if end == -1:
                pos = start
                break

            pos = end + len(self.tool_call_end_token)

        self._sent_content_idx = pos
        return "".join(content_parts) or None

    def _extract_tool_call_regions(
        self, current_text: str, *, include_partial: bool
    ) -> list[tuple[str, str, bool]]:
        """Extract visible Gemma4 tool-call regions from accumulated text."""
        tool_calls: list[tuple[str, str, bool]] = []
        pos = 0

        while True:
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                break

            body_start = start + len(self.tool_call_start_token)
            end = current_text.find(self.tool_call_end_token, body_start)
            if end == -1:
                if not include_partial:
                    break
                body = current_text[body_start:]
                overlap = partial_tag_overlap(body, self.tool_call_end_token)
                if overlap:
                    body = body[:-overlap]
                is_complete = False
                pos = len(current_text)
            else:
                body = current_text[body_start:end]
                is_complete = True
                pos = end + len(self.tool_call_end_token)

            parsed = self._parse_tool_call_body(body)
            if parsed is not None:
                func_name, raw_args_str = parsed
                tool_calls.append((func_name, raw_args_str, is_complete))

            if not is_complete:
                break

        return tool_calls

    def _parse_tool_call_body(self, body: str) -> tuple[str, str] | None:
        """Parse ``call:name{args}`` from a raw Gemma4 tool-call body."""
        if not body.startswith("call:"):
            return None

        func_part = body[len("call:") :]
        open_brace = func_part.find("{")
        if open_brace == -1:
            return None

        func_name = func_part[:open_brace].strip()
        if not func_name:
            return None

        args_start = open_brace + 1
        args_end = self._find_outer_args_end(func_part, args_start)
        raw_args_str = (
            func_part[args_start:args_end]
            if args_end is not None
            else func_part[args_start:]
        )
        return func_name, raw_args_str

    def _find_outer_args_end(self, text: str, args_start: int) -> int | None:
        """Return the index of the outer Gemma4 argument brace, if present."""
        depth = 1
        i = args_start
        text_len = len(text)

        while i < text_len:
            if text[i:].startswith(STRING_DELIM):
                i += len(STRING_DELIM)
                next_delim = text.find(STRING_DELIM, i)
                if next_delim == -1:
                    return None
                i = next_delim + len(STRING_DELIM)
                continue

            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1

        return None

    def _ensure_tool_state(self, index: int) -> None:
        """Ensure streaming state exists for tool-call index."""
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")
        self.current_tool_id = max(self.current_tool_id, index)

    def _compute_args_diff(
        self, index: int, raw_args_str: str, is_complete: bool
    ) -> str | None:
        """Return new argument JSON not yet sent for tool-call index."""
        try:
            current_args = _parse_gemma4_args(raw_args_str, partial=not is_complete)
        except Exception:
            logger.debug(
                "Could not parse partial Gemma4 args yet: %s",
                raw_args_str[:100],
            )
            return None

        if not current_args and not is_complete:
            return None

        current_args_json = json.dumps(current_args, ensure_ascii=False)
        next_json = (
            current_args_json
            if is_complete
            else self._stable_partial_json_prefix(current_args_json)
        )

        prev_streamed = self.streamed_args_for_tool[index]
        if not next_json or next_json == prev_streamed:
            return None

        if not next_json.startswith(prev_streamed):
            prefix = find_common_prefix(prev_streamed, next_json)
            self.streamed_args_for_tool[index] = prefix
            logger.debug(
                "Skipping Gemma4 argument delta for tool %d because parsed "
                "JSON no longer extends the streamed prefix.",
                index,
            )
            return None

        diff = next_json[len(prev_streamed) :]
        if not diff:
            return None

        self.streamed_args_for_tool[index] = next_json
        self.prev_tool_call_arr[index]["arguments"] = current_args
        return diff

    @staticmethod
    def _stable_partial_json_prefix(json_text: str) -> str:
        """Trim closing chars that may move while a Gemma4 call is partial."""
        return json_text.rstrip('}"]<|\\>')
