# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import regex as re
from openai.types.responses import ToolChoiceFunction

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
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import partial_tag_overlap

_TOOL_CALL_MARKER = "**Tool Call:**"
_LEGACY_TOOL_CALL_MARKER = "<tool_call>"
_JSON_FENCE_START = "```json"
_FENCED_JSON_RE = re.compile(
    r"\*\*Tool Call:\*\*\s*```json\s*(?P<json>.*?)\s*```",
    re.DOTALL,
)
_STANDALONE_FENCED_JSON_RE = re.compile(
    r"```json\s*(?P<json>.*?)\s*```",
    re.DOTALL,
)


@dataclass(frozen=True)
class _ToolCallMatch:
    start: int
    payload: dict[str, Any]


class RWKVToolParser(ToolParser):
    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._sent_content_idx = 0

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(
                tc, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            ):
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)

    def _allowed_tool_names(self) -> set[str]:
        names = set[str]()
        for tool in self.tools:
            function = getattr(tool, "function", tool)
            name = getattr(function, "name", None)
            if name is not None:
                names.add(name)
        return names

    @staticmethod
    def _loads_mapping(payload: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _normalize_legacy_arguments(
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if "path" not in arguments and isinstance(arguments.get("filePath"), str):
            arguments = {**arguments, "path": arguments["filePath"]}

        if name == "read" and "path" in arguments:
            read_arguments = {"path": arguments["path"]}
            if "offset" in arguments:
                read_arguments["offset"] = arguments["offset"]
            if "limit" in arguments:
                read_arguments["limit"] = arguments["limit"]
            return read_arguments

        if name == "edit":
            path = arguments.get("path")
            edits = arguments.get("edits")
            if not isinstance(path, str) and isinstance(edits, list):
                for edit in edits:
                    if not isinstance(edit, dict):
                        continue
                    nested_path = edit.get("path", edit.get("filePath"))
                    if isinstance(nested_path, str):
                        path = nested_path
                        break

            if not isinstance(path, str):
                return arguments

            edit_arguments: dict[str, Any] = {"path": path}
            if isinstance(edits, list):
                normalized_edits = []
                for edit in edits:
                    if not isinstance(edit, dict):
                        continue
                    if {
                        "oldText",
                        "newText",
                    } <= edit.keys():
                        normalized_edits.append(
                            {
                                "oldText": edit["oldText"],
                                "newText": edit["newText"],
                            }
                        )
                if normalized_edits:
                    edit_arguments["edits"] = normalized_edits
            elif {
                "oldText",
                "newText",
            } <= arguments.keys():
                edit_arguments["edits"] = [
                    {
                        "oldText": arguments["oldText"],
                        "newText": arguments["newText"],
                    }
                ]
            return edit_arguments

        return arguments

    def _parse_tool_call_payload(self, payload: str) -> dict[str, Any] | None:
        parsed = self._loads_mapping(payload)
        if parsed is None:
            return None
        if not isinstance(parsed.get("name"), str):
            return None
        if "arguments" not in parsed:
            return None
        if isinstance(parsed["arguments"], str):
            arguments = self._loads_mapping(parsed["arguments"])
            if arguments is None:
                return None
            parsed["arguments"] = arguments
        if not isinstance(parsed["arguments"], dict):
            return None
        parsed["arguments"] = self._normalize_legacy_arguments(
            parsed["name"],
            parsed["arguments"],
        )
        allowed_names = self._allowed_tool_names()
        if allowed_names and parsed["name"] not in allowed_names:
            return None
        return parsed

    def _iter_tool_call_matches(
        self,
        text: str,
    ) -> Sequence[_ToolCallMatch]:
        if _TOOL_CALL_MARKER in text:
            matches: list[_ToolCallMatch] = []
            for match in _FENCED_JSON_RE.finditer(text):
                payload = self._parse_tool_call_payload(match.group("json"))
                if payload is None:
                    return []
                matches.append(_ToolCallMatch(match.start(), payload))
            return matches

        if not self.tools:
            return []

        matches = []
        marker_end = 0
        while (marker_start := text.find(_LEGACY_TOOL_CALL_MARKER, marker_end)) != -1:
            payload_start = marker_start + len(_LEGACY_TOOL_CALL_MARKER)
            line_start = text.find("{", payload_start)
            if line_start == -1:
                break
            line_end = text.find("\n", line_start)
            if line_end == -1:
                break
            payload = self._parse_tool_call_payload(text[line_start:line_end].strip())
            if payload is not None:
                matches.append(_ToolCallMatch(marker_start, payload))
            marker_end = line_end

        for match in _STANDALONE_FENCED_JSON_RE.finditer(text):
            payload = self._parse_tool_call_payload(match.group("json"))
            if payload is not None:
                matches.append(_ToolCallMatch(match.start(), payload))
        matches.sort(key=lambda match: match.start)
        return matches

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        del request
        matches = self._iter_tool_call_matches(model_output)
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for match in matches:
            payload = match.payload
            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=payload["name"],
                        arguments=json.dumps(payload["arguments"], ensure_ascii=False),
                    ),
                )
            )

        content = model_output[: matches[0].start]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content if content else None,
        )

    def _extract_content_delta(self, current_text: str) -> str | None:
        marker_starts = [
            current_text.find(_TOOL_CALL_MARKER),
            current_text.find(_LEGACY_TOOL_CALL_MARKER),
        ]
        if self.tools:
            marker_starts.append(current_text.find(_JSON_FENCE_START))
        marker_start = min(
            (start for start in marker_starts if start != -1),
            default=-1,
        )
        if marker_start == -1:
            overlap = partial_tag_overlap(current_text, _TOOL_CALL_MARKER)
            overlap = max(
                overlap,
                partial_tag_overlap(current_text, _LEGACY_TOOL_CALL_MARKER),
            )
            if self.tools:
                overlap = max(
                    overlap, partial_tag_overlap(current_text, _JSON_FENCE_START)
                )
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = marker_start

        if sendable_idx <= self._sent_content_idx:
            return None
        content = current_text[self._sent_content_idx : sendable_idx]
        self._sent_content_idx = sendable_idx
        return content

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
        del previous_text, delta_text, previous_token_ids, current_token_ids
        del delta_token_ids, request

        content = self._extract_content_delta(current_text)
        tool_deltas: list[DeltaToolCall] = []

        for index, match in enumerate(self._iter_tool_call_matches(current_text)):
            if index < len(self.prev_tool_call_arr):
                continue

            payload = match.payload
            arguments = json.dumps(payload["arguments"], ensure_ascii=False)
            self.prev_tool_call_arr.append(
                {
                    "name": payload["name"],
                    "arguments": arguments,
                }
            )
            self.streamed_args_for_tool.append(arguments)
            tool_deltas.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=payload["name"],
                        arguments=arguments,
                    ),
                )
            )

        if content or tool_deltas:
            return DeltaMessage(content=content, tool_calls=tool_deltas)
        return None
