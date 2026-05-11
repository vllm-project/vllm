# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

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
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import partial_tag_overlap

logger = init_logger(__name__)


class NemotronNanoV2ToolParser(ToolParser):
    """Tool parser for Nemotron Nano v2 models that emit <TOOLCALL> JSON."""

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.tool_call_start_token = "<TOOLCALL>"
        self.tool_call_end_token = "</TOOLCALL>"
        self.tool_call_regex = re.compile(
            rf"{self.tool_call_start_token}(.*?){self.tool_call_end_token}",
            re.DOTALL,
        )
        self._sent_content_idx = 0
        self._tool_args_emitted: list[bool] = []

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    @staticmethod
    def _normalize_tool_call_payload(payload: str) -> list[dict[str, Any]]:
        payload = payload.strip()
        if not payload.startswith("["):
            payload = "[" + payload
        if not payload.endswith("]"):
            payload = payload + "]"

        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []

    @staticmethod
    def _serialize_arguments(arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments, ensure_ascii=False)

    @staticmethod
    def _strip_trailing_auto_closers(chunk: str) -> str:
        idx = len(chunk)
        while idx > 0 and chunk[idx - 1] in " \t\r\n}]":
            idx -= 1
        while idx > 0 and chunk[idx - 1] == '"':
            if idx - 2 >= 0 and chunk[idx - 2] == "\\":
                break
            idx -= 1
        return chunk[:idx]

    @staticmethod
    def _common_prefix_len(left: str, right: str) -> int:
        max_len = min(len(left), len(right))
        idx = 0
        while idx < max_len and left[idx] == right[idx]:
            idx += 1
        return idx

    def _compute_arguments_delta(self, arguments: Any, end_of_call: bool) -> str:
        if self.current_tool_id < 0:
            return ""

        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")
        while len(self._tool_args_emitted) <= self.current_tool_id:
            self._tool_args_emitted.append(False)

        cur_arguments = self._serialize_arguments(arguments)
        streamed_prefix = self.streamed_args_for_tool[self.current_tool_id]
        emitted_any = self._tool_args_emitted[self.current_tool_id]

        lcp_len = self._common_prefix_len(cur_arguments, streamed_prefix)
        if lcp_len != len(streamed_prefix):
            streamed_prefix = streamed_prefix[:lcp_len]
            self.streamed_args_for_tool[self.current_tool_id] = streamed_prefix

        arguments_delta = cur_arguments[lcp_len:]
        if not arguments_delta:
            return ""

        if not end_of_call:
            arguments_delta = self._strip_trailing_auto_closers(arguments_delta)

        if (
            not emitted_any
            and not end_of_call
            and arguments_delta
            and arguments_delta.endswith("}")
        ):
            arguments_delta = arguments_delta[:-1]
            if arguments_delta.endswith('"'):
                arguments_delta = arguments_delta[:-1]

        return arguments_delta

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            payloads = self.tool_call_regex.findall(model_output)
            tool_calls: list[ToolCall] = []
            for payload in payloads:
                for raw_tool_call in self._normalize_tool_call_payload(payload):
                    try:
                        tool_calls.append(
                            ToolCall(
                                type="function",
                                function=FunctionCall(
                                    name=raw_tool_call["name"],
                                    arguments=self._serialize_arguments(
                                        raw_tool_call["arguments"]
                                    ),
                                ),
                            )
                        )
                    except Exception:
                        continue

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            content = model_output[: model_output.find(self.tool_call_start_token)]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
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
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []
            self._tool_args_emitted = []
            self._sent_content_idx = 0

        start_idx = current_text.find(self.tool_call_start_token)
        if start_idx == -1:
            overlap = partial_tag_overlap(current_text, self.tool_call_start_token)
            sendable_idx = len(current_text) - overlap
            if sendable_idx > self._sent_content_idx:
                content = current_text[self._sent_content_idx : sendable_idx]
                self._sent_content_idx = sendable_idx
                return DeltaMessage(content=content)
            return None

        content_delta: str | None = None
        if self._sent_content_idx < start_idx:
            content_delta = current_text[self._sent_content_idx : start_idx]
            self._sent_content_idx = start_idx

        payload_start = start_idx + len(self.tool_call_start_token)
        payload_end = current_text.find(self.tool_call_end_token, payload_start)
        end_of_call = payload_end != -1
        payload = current_text[
            payload_start : payload_end if end_of_call else len(current_text)
        ]
        if not payload.strip():
            return None

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            parsed_tool_calls = partial_json_parser.loads(payload, flags)
        except (
            partial_json_parser.core.exceptions.MalformedJSON,
            json.JSONDecodeError,
            ValueError,
        ):
            return None

        if isinstance(parsed_tool_calls, dict):
            parsed_tool_calls = [parsed_tool_calls]
        if not isinstance(parsed_tool_calls, list) or not parsed_tool_calls:
            return None

        if self.current_tool_id < 0:
            self.current_tool_id = 0
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")
            self._tool_args_emitted.append(False)

        tool_call_deltas: list[DeltaToolCall] = []
        while self.current_tool_id < len(parsed_tool_calls):
            current_tool_call = parsed_tool_calls[self.current_tool_id]
            if not isinstance(current_tool_call, dict):
                break

            call_complete = end_of_call or self.current_tool_id + 1 < len(
                parsed_tool_calls
            )

            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if not function_name:
                    break

                arguments_delta = ""
                if "arguments" in current_tool_call:
                    arguments_delta = self._compute_arguments_delta(
                        current_tool_call["arguments"], call_complete
                    )
                    if arguments_delta:
                        self.streamed_args_for_tool[self.current_tool_id] += (
                            arguments_delta
                        )
                        self._tool_args_emitted[self.current_tool_id] = True

                self.current_tool_name_sent = True
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=make_tool_call_id(),
                        type="function",
                        function=DeltaFunctionCall(
                            name=function_name,
                            arguments=arguments_delta or None,
                        ),
                    )
                )
            elif "arguments" in current_tool_call:
                arguments_delta = self._compute_arguments_delta(
                    current_tool_call["arguments"], call_complete
                )
                if arguments_delta:
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                    self._tool_args_emitted[self.current_tool_id] = True
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=arguments_delta),
                        )
                    )
            elif not call_complete:
                break

            if self.current_tool_id + 1 >= len(parsed_tool_calls):
                break

            self.current_tool_id += 1
            self.current_tool_name_sent = False
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")
            while len(self._tool_args_emitted) <= self.current_tool_id:
                self._tool_args_emitted.append(False)

        if content_delta is not None or tool_call_deltas:
            return DeltaMessage(
                content=content_delta,
                tool_calls=tool_call_deltas or None,
            )
        return None
