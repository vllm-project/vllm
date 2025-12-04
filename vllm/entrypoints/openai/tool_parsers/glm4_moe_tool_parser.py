# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class Glm4MoeModelToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

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
        self._buffer = ""

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        def _is_string_type(
            tool_name: str,
            arg_name: str,
            tools: list[ChatCompletionToolsParam] | None,
        ) -> bool:
            if tools is None:
                return False
            for tool in tools:
                if tool.function.name == tool_name:
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

        def _deserialize(value: str) -> Any:
            try:
                return json.loads(value)
            except Exception:
                pass

            try:
                return ast.literal_eval(value)
            except Exception:
                pass
            return value

        matched_tool_calls = self.func_call_regex.findall(model_output)
        logger.debug("model_output: %s", model_output)
        try:
            tool_calls = []
            for match in matched_tool_calls:
                tc_detail = self.func_detail_regex.search(match)
                tc_name = tc_detail.group(1)
                tc_args = tc_detail.group(2)
                pairs = self.func_arg_regex.findall(tc_args)
                arg_dct = {}
                for key, value in pairs:
                    arg_key = key.strip()
                    arg_val = value.strip()
                    if not _is_string_type(tc_name, arg_key, request.tools):
                        arg_val = _deserialize(arg_val)
                    logger.debug("arg_key = %s, arg_val = %s", arg_key, arg_val)
                    arg_dct[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tc_name, arguments=json.dumps(arg_dct)
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
                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=content
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
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
        self._buffer += delta_text
        cur_text = self._buffer
        start_idx = cur_text.find(self.tool_call_start_token)
        if start_idx == -1:
            self._buffer = ""
            if self.current_tool_id > 0:
                cur_text = ""
            return DeltaMessage(content=cur_text)
        logger.debug("cur_text = %s", cur_text)
        end_idx = cur_text.find(self.tool_call_end_token)
        if end_idx != -1:
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            extracted_tool_calls = self.extract_tool_calls(
                cur_text[: end_idx + len(self.tool_call_end_token)], request
            )

            if len(extracted_tool_calls.tool_calls) == 0:
                logger.warning("Failed to extract any tool calls.")
                return None
            tool_call = extracted_tool_calls.tool_calls[0]
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
            self.streamed_args_for_tool[self.current_tool_id] = (
                tool_call.function.arguments
            )
            delta = DeltaMessage(
                content=extracted_tool_calls.content,
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=DeltaFunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                ],
            )
            self.current_tool_id += 1
            self._buffer = cur_text[end_idx + len(self.tool_call_end_token) :]
            return delta

        self._buffer = cur_text[start_idx:]
        return DeltaMessage(content=cur_text[:start_idx])
