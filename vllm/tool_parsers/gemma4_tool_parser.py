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
from dataclasses import dataclass

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
from vllm.tool_parsers.utils import find_common_prefix, partial_tag_overlap

logger = init_logger(__name__)


# Gemma4 special tokens for tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
TOOL_RESPONSE = "<|tool_response>"
STRING_DELIM = '<|"|>'
_NUMBER_LITERAL_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)
_INT_LITERAL_RE = re.compile(r"^[+-]?\d+$")


@dataclass
class _StreamingToolCall:
    name: str
    raw_arguments: str
    complete: bool


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

    # Number (int or float). Gemma4 follows JSON-like numeric literals, and
    # exponent-form values such as 1e3 must remain numeric, not bare strings.
    if _NUMBER_LITERAL_RE.match(value_str):
        if _INT_LITERAL_RE.match(value_str):
            return int(value_str)
        return float(value_str)

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    return value_str


def _is_unstable_partial_bare_value(value_str: str) -> bool:
    value_str = value_str.strip()
    if not value_str:
        return True

    lower = value_str.lower()
    for literal in ("true", "false", "null", "none", "nil"):
        if literal.startswith(lower) and lower != literal:
            return True

    # Numeric literals are type-unstable while ending in decimal/exponent
    # punctuation. This mirrors llama.cpp's "maybe number" partial handling.
    if value_str in {"+", "-"}:
        return True
    return value_str[-1] in {".", "e", "E", "+", "-"} and any(
        char.isdigit() for char in value_str
    )


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
                if _is_unstable_partial_bare_value(raw_val):
                    # Decimal/exponent digits may still arrive: "108." can
                    # become "108.2", and "1e" can become "1e3".
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
                if _is_unstable_partial_bare_value(raw_val):
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
        self.tool_response_token = TOOL_RESPONSE

        # Token IDs
        self.tool_call_start_token_id = self.vocab.get(TOOL_CALL_START)
        self.tool_call_end_token_id = self.vocab.get(TOOL_CALL_END)
        self.tool_response_token_id = self.vocab.get(TOOL_RESPONSE)

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

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new request."""
        # Keep the public backfill fields empty. Gemma4 does its own
        # complete-call diffing, and the generic serving-layer terminal
        # backfill cannot safely replay multiple Gemma4 calls from one delta.
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self._streaming_tool_states: list[dict] = []
        self._streamed_args_for_tool: list[str] = []
        self._streaming_text = ""
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
    # Delta repair for multi-token/speculative special sequences
    # ------------------------------------------------------------------

    def _repair_delta_text_from_token_ids(
        self, delta_text: str, delta_token_ids: Sequence[int]
    ) -> str:
        """Restore Gemma4 tool-call text when text deltas omit token text.

        Speculative/MTP streaming can surface tool-call token IDs while the
        corresponding ``delta_text`` is empty, especially around Gemma4 special
        tokens. In that case the accumulated text never contains the closing
        ``<tool_call|>`` marker, so streaming parsers cannot safely expose the
        completed call. Only repair the Gemma4 tool-call path; normal content
        continues to use the engine-provided text delta.
        """
        special_token_ids = {
            token_id
            for token_id in (
                self.tool_call_start_token_id,
                self.tool_call_end_token_id,
                self.tool_response_token_id,
            )
            if token_id is not None
        }
        has_special_token = bool(special_token_ids) and any(
            token_id in special_token_ids for token_id in delta_token_ids
        )
        if not delta_token_ids or (
            not has_special_token
            and (delta_text or not self._inside_tool_call_text())
        ):
            return delta_text

        try:
            decoded = self.model_tokenizer.decode(
                list(delta_token_ids), skip_special_tokens=False
            )
        except TypeError:
            decoded = self.model_tokenizer.decode(list(delta_token_ids))
        except Exception:
            logger.debug("Could not decode Gemma4 delta token ids", exc_info=True)
            return delta_text

        if not isinstance(decoded, str) or not decoded:
            return delta_text

        # <|tool_response> is a stop/control marker for the next turn, not part
        # of the assistant tool call payload. Keep the tool-call end marker that
        # may precede it, but never stream the response marker as content.
        decoded = decoded.replace(self.tool_response_token, "")
        if not decoded:
            return delta_text

        missing_start = (
            self.tool_call_start_token in decoded
            and self.tool_call_start_token not in delta_text
        )
        missing_end = (
            self.tool_call_end_token in decoded
            and self.tool_call_end_token not in delta_text
        )
        if not delta_text or missing_start or missing_end:
            return decoded
        return delta_text

    def _inside_tool_call_text(self) -> bool:
        last_start = self._streaming_text.rfind(self.tool_call_start_token)
        if last_start == -1:
            return False
        last_end = self._streaming_text.rfind(self.tool_call_end_token)
        return last_end < last_start

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
        # Keep a Gemma4-local accumulated text buffer. Speculative/MTP streaming
        # may provide token IDs with empty text deltas for tool-call spans; the
        # serving-layer text accumulator cannot represent those repaired deltas.
        if not previous_text and not previous_token_ids:
            self._reset_streaming_state()
        previous_text = self._streaming_text
        delta_text = self._repair_delta_text_from_token_ids(
            delta_text, delta_token_ids
        )
        current_text = previous_text + delta_text
        self._streaming_text = current_text

        try:
            return self._extract_streaming(
                current_text=current_text,
                request=request,
            )
        except Exception:
            logger.exception("Error in Gemma4 streaming tool call extraction")
            return None

    def _extract_streaming(
        self,
        current_text: str,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Parse accumulated text into structured tool calls, then emit diffs."""
        tool_calls = self._parse_streaming_tool_calls(current_text)
        if request.parallel_tool_calls is False:
            tool_calls = tool_calls[:1]

        content = self._extract_content(current_text)
        tool_call_deltas = self._diff_streaming_tool_calls(tool_calls)

        if content or tool_call_deltas:
            return DeltaMessage(content=content, tool_calls=tool_call_deltas)
        return None

    def _parse_streaming_tool_calls(self, text: str) -> list[_StreamingToolCall]:
        """Extract complete calls plus the active partial call from accumulated text."""
        tool_calls: list[_StreamingToolCall] = []
        pos = 0

        while True:
            start = text.find(self.tool_call_start_token, pos)
            if start == -1:
                break

            body_start = start + len(self.tool_call_start_token)
            if not text.startswith("call:", body_start):
                pos = body_start
                continue

            name_start = body_start + len("call:")
            brace = text.find("{", name_start)
            next_start = text.find(self.tool_call_start_token, body_start)
            end_before_brace = text.find(self.tool_call_end_token, body_start)
            if (
                brace == -1
                or (next_start != -1 and next_start < brace)
                or (end_before_brace != -1 and end_before_brace < brace)
            ):
                break

            name = text[name_start:brace].strip()
            if not name:
                break

            end = text.find(self.tool_call_end_token, brace + 1)
            complete = end != -1
            if complete:
                raw_arguments = text[brace + 1 : end]
                pos = end + len(self.tool_call_end_token)
            else:
                raw_arguments = text[brace + 1 :]
                next_partial_start = raw_arguments.find(self.tool_call_start_token)
                if next_partial_start != -1:
                    raw_arguments = raw_arguments[:next_partial_start]
                pos = len(text)

            if raw_arguments.endswith("}"):
                raw_arguments = raw_arguments[:-1]

            tool_calls.append(
                _StreamingToolCall(
                    name=name,
                    raw_arguments=raw_arguments,
                    complete=complete,
                )
            )

            if not complete:
                break

        return tool_calls

    def _diff_streaming_tool_calls(
        self, tool_calls: list[_StreamingToolCall]
    ) -> list[DeltaToolCall]:
        deltas: list[DeltaToolCall] = []

        for index, tool_call in enumerate(tool_calls):
            # Do not expose partial tool-call arguments. If generation stops
            # before Gemma4 emits <tool_call|>, streaming clients
            # cannot retract already-streamed malformed JSON. Buffering until
            # the complete tool-call marker keeps streamed tool calls valid.
            if not tool_call.complete:
                continue

            self._ensure_streaming_tool_state(index, tool_call.name)
            state = self._streaming_tool_states[index]
            emit_header = not state.get("name_sent", False)
            arguments_diff = self._compute_arguments_diff(index, tool_call)

            if not emit_header and arguments_diff is None:
                continue

            deltas.append(
                DeltaToolCall(
                    index=index,
                    id=state["id"] if emit_header else None,
                    type="function" if emit_header else None,
                    function=DeltaFunctionCall(
                        name=tool_call.name if emit_header else None,
                        arguments=arguments_diff if arguments_diff is not None else "",
                    ).model_dump(exclude_none=True),
                )
            )
            if emit_header:
                state["name_sent"] = True

        return deltas

    def _ensure_streaming_tool_state(self, index: int, name: str) -> None:
        while len(self._streaming_tool_states) <= index:
            self._streaming_tool_states.append({})
        while len(self._streamed_args_for_tool) <= index:
            self._streamed_args_for_tool.append("")

        state = self._streaming_tool_states[index]
        state.setdefault("id", make_tool_call_id())
        state.setdefault("name", name)
        state.setdefault("arguments", {})

    def _compute_arguments_diff(
        self, index: int, tool_call: _StreamingToolCall
    ) -> str | None:
        arguments_json = self._arguments_json_for_streaming(tool_call)
        if arguments_json is None:
            return None

        prev_streamed = self._streamed_args_for_tool[index]
        if arguments_json == prev_streamed:
            return None

        if prev_streamed:
            prefix = find_common_prefix(prev_streamed, arguments_json)
            if len(prefix) < len(prev_streamed):
                self._streamed_args_for_tool[index] = prefix
                return None
            diff = arguments_json[len(prev_streamed) :]
        else:
            diff = arguments_json

        if not diff:
            return None

        self._streamed_args_for_tool[index] = arguments_json
        self._streaming_tool_states[index]["arguments"] = json.loads(arguments_json)
        return diff

    def _arguments_json_for_streaming(
        self, tool_call: _StreamingToolCall
    ) -> str | None:
        try:
            arguments = _parse_gemma4_args(
                tool_call.raw_arguments,
                partial=not tool_call.complete,
            )
        except Exception:
            logger.debug(
                "Could not parse Gemma4 streaming args yet: %s",
                tool_call.raw_arguments[:100],
            )
            return None

        if not arguments and not tool_call.complete:
            return None

        return json.dumps(arguments, ensure_ascii=False)

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
                self.tool_call_end_token,
                start + len(self.tool_call_start_token),
            )
            if end == -1:
                pos = start
                break
            pos = end + len(self.tool_call_end_token)

        self._sent_content_idx = pos
        return "".join(content_parts) or None
