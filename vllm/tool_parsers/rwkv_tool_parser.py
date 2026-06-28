# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
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
_FENCED_JSON_RE = re.compile(
    r"\*\*Tool Call:\*\*\s*```json\s*(?P<json>.*?)\s*```",
    re.DOTALL,
)


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

    @staticmethod
    def _tool_name(tool: Tool) -> str | None:
        function = getattr(tool, "function", tool)
        return getattr(function, "name", None)

    def _allowed_tool_names(self) -> set[str]:
        return {
            name for tool in self.tools if (name := self._tool_name(tool)) is not None
        }

    def _parse_tool_call_payload(self, payload: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        if not isinstance(parsed.get("name"), str):
            return None
        if "arguments" not in parsed:
            return None
        allowed_names = self._allowed_tool_names()
        if allowed_names and parsed["name"] not in allowed_names:
            return None
        return parsed

    @staticmethod
    def _tool_call_from_payload(payload: dict[str, Any]) -> ToolCall:
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=payload["name"],
                arguments=json.dumps(payload["arguments"], ensure_ascii=False),
            ),
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        del request
        if _TOOL_CALL_MARKER not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        matches = list(_FENCED_JSON_RE.finditer(model_output))
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        for match in matches:
            payload = self._parse_tool_call_payload(match.group("json"))
            if payload is None:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )
            tool_calls.append(self._tool_call_from_payload(payload))

        content = model_output[: matches[0].start()]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content if content else None,
        )

    def _extract_content_delta(self, current_text: str) -> str | None:
        marker_start = current_text.find(_TOOL_CALL_MARKER)
        if marker_start == -1:
            overlap = partial_tag_overlap(current_text, _TOOL_CALL_MARKER)
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

        for index, match in enumerate(_FENCED_JSON_RE.finditer(current_text)):
            if index < len(self.prev_tool_call_arr):
                continue
            payload = self._parse_tool_call_payload(match.group("json"))
            if payload is None:
                continue

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
