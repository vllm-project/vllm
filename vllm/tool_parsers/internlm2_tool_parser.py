# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    is_complete_json,
    partial_tag_overlap,
)

logger = init_logger(__name__)


class Internlm2ToolParser(ToolParser):
    TOOL_CALL_START_TOKEN = "<|action_start|><|plugin|>"
    TOOL_CALL_END_TOKEN = "<|action_end|>"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.position = 0

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because internlm use the special
            # tokens to indicate the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        return request

    def get_arguments(self, obj):
        if "parameters" in obj:
            return obj.get("parameters")
        elif "arguments" in obj:
            return obj.get("arguments")
        return None

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent non-tool-call text, or None.

        Holds back any suffix that could be a partial opening marker.
        """
        if self.TOOL_CALL_START_TOKEN not in current_text:
            overlap = partial_tag_overlap(current_text, self.TOOL_CALL_START_TOKEN)
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = current_text.index(self.TOOL_CALL_START_TOKEN)

        if sendable_idx > self.position:
            content = current_text[self.position : sendable_idx]
            self.position = sendable_idx
            return content
        return None

    def _extract_tool_call(self, current_text: str) -> tuple[str, bool] | None:
        """Extract the action payload and completion status, or None."""
        if self.TOOL_CALL_START_TOKEN not in current_text:
            return None
        start_idx = current_text.index(self.TOOL_CALL_START_TOKEN)
        after_start = current_text[start_idx + len(self.TOOL_CALL_START_TOKEN) :]
        if self.TOOL_CALL_END_TOKEN in after_start:
            body = after_start.split(self.TOOL_CALL_END_TOKEN)[0]
            return body.strip(), True
        else:
            overlap = partial_tag_overlap(after_start, self.TOOL_CALL_END_TOKEN)
            raw = after_start[:-overlap] if overlap else after_start
            raw = raw.strip()
            is_comp = is_complete_json(raw) if raw else False
            return raw, is_comp

    @staticmethod
    def _extract_tool_name(tc_json: str) -> str | None:
        """Extract tool name, or None if the name is not yet complete."""
        match = re.search(r'"name"\s*:\s*"([^"]+)"', tc_json)
        return match.group(1) if match else None

    @staticmethod
    def _extract_tool_args(tc_json: str, is_complete: bool) -> str | None:
        """Extract tool arguments from the tool call JSON string."""
        match = re.search(r'"(?:parameters|arguments)"\s*:\s*', tc_json)
        if not match:
            return None
        raw = tc_json[match.end() :]
        if is_complete:
            raw = raw.rstrip()
            if raw.endswith("}"):
                raw = raw[:-1].rstrip()
        return raw

    def _compute_args_diff(self, tc_json: str, is_complete: bool) -> str | None:
        """Compute unsent arguments diff for the active tool call."""
        args = self._extract_tool_args(tc_json, is_complete)
        if args is None:
            return None
        streamed = self.streamed_args_for_tool[self.current_tool_id]
        if len(args) <= len(streamed):
            return None
        diff = args[len(streamed) :]
        self.streamed_args_for_tool[self.current_tool_id] = args
        return diff

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
            self._sent_content_idx = 0
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []

        try:
            # 1. Check for unsent plain content before tool call
            content = self._extract_content(current_text)
            if content:
                return DeltaMessage(content=content)

            # 2. Extract tool call body
            tc_info = self._extract_tool_call(current_text)
            if not tc_info:
                return None

            tc_json, is_complete = tc_info
            if not tc_json:
                return None

            tool_call_deltas: list[DeltaToolCall] = []

            # 3. If tool name hasn't been sent yet, send it
            if not self.current_tool_name_sent:
                name = self._extract_tool_name(tc_json)
                if not name:
                    return None
                self.current_tool_id += 1
                self.current_tool_name_sent = True
                self.streamed_args_for_tool.append("")

                args = self._extract_tool_args(tc_json, is_complete)
                if args:
                    self.streamed_args_for_tool[self.current_tool_id] = args
                    fn_call = DeltaFunctionCall(name=name, arguments=args).model_dump(
                        exclude_none=True
                    )
                else:
                    fn_call = DeltaFunctionCall(name=name).model_dump(exclude_none=True)

                tool_call_deltas.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=fn_call,
                    )
                )
                return DeltaMessage(tool_calls=tool_call_deltas)

            # 4. Stream incremental arguments
            diff = self._compute_args_diff(tc_json, is_complete)
            if diff:
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=diff).model_dump(
                            exclude_none=True
                        ),
                    )
                )
                return DeltaMessage(tool_calls=tool_call_deltas)

            return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        text = model_output
        tools = self.tools
        if self.TOOL_CALL_START_TOKEN in text:
            text, action = text.split(self.TOOL_CALL_START_TOKEN, 1)
            action = action.split(self.TOOL_CALL_END_TOKEN.strip())[0]
            json_start = action.find("{")
            if json_start != -1:
                action = action[json_start:]
            try:
                action_dict = json.loads(action)
                name = action_dict["name"]
                parameters = json.dumps(
                    action_dict.get("parameters", action_dict.get("arguments", {})),
                    ensure_ascii=False,
                )

                if tools and name not in [t.function.name for t in tools]:
                    return ExtractedToolCallInformation(
                        tools_called=False,
                        tool_calls=[],
                        content=text if len(text) > 0 else None,
                    )

                tool_calls = [
                    ToolCall(function=FunctionCall(name=name, arguments=parameters))
                ]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=text if len(text) > 0 else None,
                )
            except Exception:
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=text if len(text) > 0 else None,
                )

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=text if len(text) > 0 else None
        )
