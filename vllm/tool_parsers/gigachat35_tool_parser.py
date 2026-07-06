# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
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

TOOL_CALLS_START = "<｜GCML｜tool_calls>"
TOOL_CALLS_END = "</｜GCML｜tool_calls>"

INVOKE_RE = re.compile(
    r'<｜GCML｜invoke\b(?P<attrs>[^>]*)>(?P<body>.*?)</｜GCML｜invoke>',
    re.DOTALL,
)
PARAMETER_RE = re.compile(
    r'<｜GCML｜parameter\b(?P<attrs>[^>]*)>(?P<body>.*?)</｜GCML｜parameter>',
    re.DOTALL,
)
ATTR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')


def _strip_trailing_eos(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("</s>"):
        return stripped[: -len("</s>")]
    return text


def _attrs(raw: str) -> dict[str, str]:
    return {key: value for key, value in ATTR_RE.findall(raw)}


def _parse_parameter_value(raw: str, is_string: bool) -> object:
    if is_string:
        return raw
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return raw


def _parse_invocations(text: str) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []
    for invoke_match in INVOKE_RE.finditer(text):
        invoke_attrs = _attrs(invoke_match.group("attrs"))
        name = invoke_attrs.get("name")
        if not name:
            continue

        args: dict[str, object] = {}
        for parameter_match in PARAMETER_RE.finditer(invoke_match.group("body")):
            parameter_attrs = _attrs(parameter_match.group("attrs"))
            parameter_name = parameter_attrs.get("name")
            if not parameter_name:
                continue
            is_string = parameter_attrs.get("string", "true").lower() == "true"
            args[parameter_name] = _parse_parameter_value(
                parameter_match.group("body"),
                is_string=is_string,
            )
        calls.append((name, args))
    return calls


class GigaChat35ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._sent_content_idx = 0

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        model_output = _strip_trailing_eos(model_output)
        start = model_output.find(TOOL_CALLS_START)
        if start == -1:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output if model_output else None,
            )

        content = model_output[:start] or None
        body_start = start + len(TOOL_CALLS_START)
        end = model_output.find(TOOL_CALLS_END, body_start)
        body = model_output[body_start:] if end == -1 else model_output[body_start:end]
        parsed_calls = _parse_invocations(body)
        if not parsed_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output if model_output else None,
            )

        tool_calls = [
            ToolCall(
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                ),
            )
            for name, args in parsed_calls
        ]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content,
        )

    def _extract_content(self, current_text: str) -> str | None:
        start = current_text.find(TOOL_CALLS_START)
        if start == -1:
            overlap = partial_tag_overlap(current_text, TOOL_CALLS_START)
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = start

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

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
        content = self._extract_content(current_text)
        start = current_text.find(TOOL_CALLS_START)
        if start == -1:
            return DeltaMessage(content=content) if content else None

        body_start = start + len(TOOL_CALLS_START)
        end = current_text.find(TOOL_CALLS_END, body_start)
        body = current_text[body_start:] if end == -1 else current_text[body_start:end]

        parsed_calls = _parse_invocations(body)
        for index, (name, args) in enumerate(parsed_calls):
            if index < len(self.prev_tool_call_arr):
                continue

            args_text = json.dumps(args, ensure_ascii=False)
            self.prev_tool_call_arr.append(
                {
                    "name": name,
                    "arguments": args,
                }
            )
            self.streamed_args_for_tool.append(args_text)
            return DeltaMessage(
                content=content,
                tool_calls=[
                    DeltaToolCall(
                        index=index,
                        id=make_tool_call_id(),
                        type="function",
                        function=DeltaFunctionCall(
                            name=name,
                            arguments=args_text,
                        ).model_dump(exclude_none=True),
                    )
                ],
            )

        return DeltaMessage(content=content) if content else None
