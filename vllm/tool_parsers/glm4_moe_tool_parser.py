# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM-4 Tool Call Parser with incremental string streaming support.

This parser fixes the streaming issue reported in Issue #32829 where long string
parameters (e.g., file content with 4000+ characters of code) are buffered until
complete, causing multi-second delays before the user sees any content.

The fix streams string values incrementally as they arrive, providing a true
streaming experience for long content.
"""

import ast
import json
from collections.abc import Sequence
from typing import Any

import regex as re

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
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import partial_tag_overlap

logger = init_logger(__name__)


class Glm4MoeModelToolParser(ToolParser):
    """Tool parser for GLM-4 models with incremental string streaming.

    On every streaming call the parser re-parses ``current_text`` to find
    ``<tool_call>`` regions, builds the JSON arguments string for each tool
    call, and diffs against what was previously sent to emit only new content.
    """

    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # Stateful streaming fields
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.arg_key_start: str = "<arg_key>"
        self.arg_key_end: str = "</arg_key>"
        self.arg_val_start: str = "<arg_value>"
        self.arg_val_end: str = "</arg_value>"

        self.tool_calls_start_token = self.tool_call_start_token

        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>([^\n]*)\n(.*)</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        # Pre-compiled pattern for finding the last <arg_key>...</arg_key>
        # before a partial <arg_value> (used in _build_args_json_so_far).
        self._arg_key_pattern = re.compile(
            re.escape(self.arg_key_start) + r"(.*?)" + re.escape(self.arg_key_end),
            re.DOTALL,
        )

        # Streaming state for re-parse-and-diff approach
        self._sent_content_idx: int = 0
        self._tool_call_ids: list[str] = []

    @staticmethod
    def _deserialize(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        return value

    @staticmethod
    def _json_escape_string_content(s: str) -> str:
        """JSON-escape string content for incremental streaming.

        This escapes the content that goes INSIDE a JSON string (between quotes),
        not including the surrounding quotes themselves.
        """
        if not s:
            return ""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    @staticmethod
    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: list[Tool] | None,
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:
            if tool.function.name != tool_name:
                continue
            if tool.function.parameters is None:
                return False
            arg_type = (
                tool.function.parameters.get("properties", {})
                .get(arg_name, {})
                .get("type", None)
            )
            return arg_type == "string"
        logger.debug("No tool named '%s'.", tool_name)
        return False

    @staticmethod
    def _tools_enabled(request: ChatCompletionRequest) -> bool:
        """Return whether tool parsing should be applied for this request."""
        try:
            tools = getattr(request, "tools", None)
            tool_choice = getattr(request, "tool_choice", None)
            return bool(tools) and tool_choice != "none"
        except Exception:
            logger.exception("Failed to determine if tools are enabled.")
            return False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Adjust request parameters for tool call token handling.

        For required/named tool_choice, skip setting structured_outputs
        because GLM models output tool calls in XML format (per chat
        template).  Guided decoding would force JSON output, conflicting
        with the XML format and causing parsing failures.
        """
        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(tc, ChatCompletionNamedToolChoiceParam):
                # Do NOT call super().adjust_request() for required/named,
                # because it would set structured_outputs and force JSON
                # output via guided decoding.  GLM models use XML tool-call
                # syntax (defined in the chat template), so guided decoding
                # must be skipped to let the model output XML freely.
                # The tool_parser handles extraction from XML output.
                if request.tool_choice != "none":
                    request.skip_special_tokens = False
                return request
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure tool call tokens (<tool_call>, </tool_call>) are not skipped
            # during decoding. Even though they are not marked as special tokens,
            # setting skip_special_tokens=False ensures proper handling in
            # transformers 5.x where decoding behavior may have changed.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matched_tool_calls = self.func_call_regex.findall(model_output)
        logger.debug("model_output: %s", model_output)
        try:
            tool_calls: list[ToolCall] = []
            for match in matched_tool_calls:
                tc_detail = self.func_detail_regex.search(match)
                if not tc_detail:
                    logger.warning(
                        "Failed to parse tool call details from: %s",
                        match,
                    )
                    continue
                tc_name = tc_detail.group(1).strip()
                tc_args = tc_detail.group(2)
                pairs = self.func_arg_regex.findall(tc_args) if tc_args else []
                arg_dct: dict[str, Any] = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    arg_val = value.strip()
                    if not self._is_string_type(tc_name, arg_key, self.tools):
                        arg_val = self._deserialize(arg_val)
                    logger.debug("arg_key = %s, arg_val = %s", arg_key, arg_val)
                    arg_dct[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tc_name,
                            arguments=json.dumps(arg_dct, ensure_ascii=False),
                        ),
                    )
                )
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        else:
            if len(tool_calls) > 0:
                content: str | None = model_output[
                    : model_output.find(self.tool_calls_start_token)
                ]
                # Normalize empty/whitespace-only content to None
                if not content or not content.strip():
                    content = None
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=content
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, or None.

        Collects all text outside ``<tool_call>...</tool_call>`` regions,
        including text between consecutive tool calls.  Holds back any
        suffix that could be a partial ``<tool_call>`` tag.
        """
        # Build the "sendable index" — the furthest point we can send
        # content up to.  We scan through the text collecting segments
        # that are outside tool-call regions.
        content_segments: list[str] = []
        pos = self._sent_content_idx

        while pos < len(current_text):
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                # No more tool calls — send up to (len - partial-tag overlap)
                tail = current_text[pos:]
                overlap = partial_tag_overlap(tail, self.tool_call_start_token)
                sendable = tail[: len(tail) - overlap] if overlap else tail
                if sendable:
                    content_segments.append(sendable)
                pos = len(current_text) - overlap
                break

            # Text before this <tool_call>
            if start > pos:
                content_segments.append(current_text[pos:start])

            # Skip past the </tool_call> (or to end if incomplete)
            end = current_text.find(self.tool_call_end_token, start)
            if end != -1:
                pos = end + len(self.tool_call_end_token)
            else:
                # Incomplete tool call — nothing more to send
                pos = start
                break

        if content_segments:
            self._sent_content_idx = pos
            return "".join(content_segments)
        # Even if no content, advance past completed tool-call regions
        if pos > self._sent_content_idx:
            self._sent_content_idx = pos
        return None

    def _extract_tool_call_regions(self, text: str) -> list[tuple[str, bool]]:
        """Extract ``(inner_text, is_complete)`` for each ``<tool_call>`` region."""
        results: list[tuple[str, bool]] = []
        pos = 0
        while True:
            start = text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            inner_start = start + len(self.tool_call_start_token)
            end = text.find(self.tool_call_end_token, inner_start)
            if end != -1:
                results.append((text[inner_start:end], True))
                pos = end + len(self.tool_call_end_token)
            else:
                # Incomplete tool call — strip partial </tool_call> suffix
                raw = text[inner_start:]
                overlap = partial_tag_overlap(raw, self.tool_call_end_token)
                if overlap:
                    raw = raw[:-overlap]
                results.append((raw, False))
                break
        return results

    def _extract_tool_name_from_region(self, inner_text: str) -> str | None:
        """Extract the tool name from the beginning of a tool-call region.

        The name is everything before the first ``\\n`` or ``<arg_key>``.
        Returns ``None`` if the name hasn't fully arrived yet.
        """
        nl = inner_text.find("\n")
        ak = inner_text.find(self.arg_key_start)
        candidates = [i for i in [nl, ak] if i != -1]
        if not candidates:
            return None
        cut = min(candidates)
        name = inner_text[:cut].strip()
        return name if name else None

    def _build_args_json_so_far(
        self,
        tool_name: str,
        inner_text: str,
        is_complete: bool,
    ) -> str:
        """Build the JSON arguments string from the XML pairs seen so far.

        For complete ``<arg_key>/<arg_value>`` pairs the value is fully
        formatted.  For the last argument whose ``<arg_value>`` has been
        opened but not closed, the partial string content is included
        (JSON-escaped, with an opening ``"`` but no closing ``"``).

        The closing ``}`` is only appended when ``is_complete`` is True
        (i.e. the ``</tool_call>`` tag has arrived).
        """
        # Find all complete arg pairs
        pairs = self.func_arg_regex.findall(inner_text)

        parts: list[str] = []
        for key, value in pairs:
            key = key.strip()
            key_json = json.dumps(key, ensure_ascii=False)
            if self._is_string_type(tool_name, key, self.tools):
                # Don't strip string values — whitespace is significant
                # and must match the partial-value path for diffing.
                val_json = json.dumps(value, ensure_ascii=False)
            else:
                val_json = json.dumps(
                    self._deserialize(value.strip()), ensure_ascii=False
                )
            parts.append(f"{key_json}: {val_json}")

        # Check for a partial (incomplete) arg value
        # Find the last <arg_value> that isn't closed
        last_val_start = inner_text.rfind(self.arg_val_start)
        last_val_end = inner_text.rfind(self.arg_val_end)
        has_partial_value = last_val_start != -1 and (
            last_val_end == -1 or last_val_end < last_val_start
        )

        if has_partial_value:
            # Find the key for this partial value
            # Look for the last <arg_key>...</arg_key> before this <arg_value>
            last_key_match = None
            for m in self._arg_key_pattern.finditer(inner_text[:last_val_start]):
                last_key_match = m

            if last_key_match:
                partial_key = last_key_match.group(1).strip()
                partial_content_start = last_val_start + len(self.arg_val_start)
                partial_content = inner_text[partial_content_start:]

                # Hold back any partial </arg_value> suffix
                overlap = partial_tag_overlap(partial_content, self.arg_val_end)
                if overlap:
                    partial_content = partial_content[:-overlap]

                key_json = json.dumps(partial_key, ensure_ascii=False)
                if is_complete:
                    # Tool call finished but </arg_value> is missing
                    # (malformed output). Treat partial as complete value
                    # so the diff naturally closes any open quotes.
                    if self._is_string_type(tool_name, partial_key, self.tools):
                        val_json = json.dumps(partial_content, ensure_ascii=False)
                    else:
                        val_json = json.dumps(
                            self._deserialize(partial_content.strip()),
                            ensure_ascii=False,
                        )
                    parts.append(f"{key_json}: {val_json}")
                elif self._is_string_type(tool_name, partial_key, self.tools):
                    escaped = self._json_escape_string_content(partial_content)
                    # Open quote but no close — more content may arrive
                    parts.append(f'{key_json}: "{escaped}')
                else:
                    # Non-string partial: include raw content, no wrapping
                    parts.append(f"{key_json}: {partial_content}")

        if not parts:
            return "{}" if is_complete else ""

        joined = "{" + ", ".join(parts)
        if is_complete:
            joined += "}"
        return joined

    def _compute_args_diff(self, index: int, args_so_far: str) -> str | None:
        """Return new argument text not yet sent for tool *index*, or None."""
        if not args_so_far or len(args_so_far) <= len(
            self.streamed_args_for_tool[index]
        ):
            return None
        diff = args_so_far[len(self.streamed_args_for_tool[index]) :]
        self.streamed_args_for_tool[index] = args_so_far
        self.prev_tool_call_arr[index]["arguments"] = args_so_far
        return diff

    def _ensure_tool_state_for(self, index: int) -> None:
        """Grow state arrays so that *index* is valid."""
        while len(self._tool_call_ids) <= index:
            self._tool_call_ids.append(
                make_tool_call_id(id_type="random", func_name=None, idx=None)
            )
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})

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
        if not self._tools_enabled(request):
            return DeltaMessage(content=delta_text) if delta_text else None

        content = self._extract_content(current_text)
        regions = self._extract_tool_call_regions(current_text)
        tool_call_deltas: list[DeltaToolCall] = []

        for i, (inner_text, is_complete) in enumerate(regions):
            self._ensure_tool_state_for(i)

            # Extract tool name
            tool_name = self._extract_tool_name_from_region(inner_text)
            if not tool_name:
                break

            # Emit tool name (once per tool call)
            if "name" not in self.prev_tool_call_arr[i]:
                self.prev_tool_call_arr[i]["name"] = tool_name
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=i,
                        id=self._tool_call_ids[i],
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_name,
                            arguments="",
                        ).model_dump(exclude_none=True),
                    )
                )

            # Build args JSON so far, diff, emit
            args_so_far = self._build_args_json_so_far(
                tool_name, inner_text, is_complete
            )
            diff = self._compute_args_diff(i, args_so_far)
            if diff:
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=i,
                        function=DeltaFunctionCall(arguments=diff).model_dump(
                            exclude_none=True
                        ),
                    )
                )

        # Update current_tool_id for serving layer compatibility
        if regions:
            self.current_tool_id = len(regions) - 1

        if content or tool_call_deltas:
            return DeltaMessage(
                content=content,
                tool_calls=tool_call_deltas,
            )
        return None
