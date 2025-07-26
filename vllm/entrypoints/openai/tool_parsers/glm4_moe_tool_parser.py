# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# code modified from deepseekv3_tool_parser.py

import ast
import json
from collections.abc import Sequence
from typing import Union

import regex as re

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("glm45")
class Glm4MoeModelToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        self.tool_calls_start_token = self.tool_call_start_token

        # Updated regex for the XML-based format
        self.tool_call_regex = re.compile(
            r"<tool_call>\s*"
            r"(?P<function_name>[^\n<]+)\s*"  # 函数名（到换行或 <）
            r"(?P<arguments>(?:\s*<arg_key>[^<]+</arg_key>\s*"
            r"<arg_value>[^<]*</arg_value>\s*)*)\s*"
            r"</tool_call>",
            re.DOTALL,
        )

        # Regex for parsing individual arguments
        self.arg_regex = re.compile(
            r"<arg_key>(?P<key>[^<]+)</arg_key>\s*<arg_value>(?P<value>[^<]*)</arg_value>",
            re.DOTALL,
        )

        # Streaming regex
        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<function_name>[^\n<]+)\s*"
            r"(?P<arguments>(?:\s*<arg_key>[^<]+</arg_key>\s*"
            r"<arg_value>[^<]*</arg_value>\s*)*)",
            re.DOTALL,
        )

        # For streaming, we also need a regex to match just the function name
        self.stream_tool_call_name_regex = re.compile(
            r"(?P<function_name>[^\n<]+)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self._buffer = ""

    def _parse_arguments(self, args_text: str) -> str:
        """Parse XML-based arguments into JSON format."""
        if not args_text or not args_text.strip():
            return "{}"

        args_dict = {}
        matches = self.arg_regex.findall(args_text)

        for key, value in matches:
            try:
                if isinstance(value, str):
                    value = json.loads(value)
            except Exception:
                pass

            try:
                if isinstance(value, str):
                    value = ast.literal_eval(value)
            except Exception:
                pass
            args_dict[key.strip()] = value

        return json.dumps(args_dict, ensure_ascii=False)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            # Find all tool calls in the output
            function_call_matches = self.tool_call_regex.findall(model_output)

            logger.debug("function_call_matches: %s", function_call_matches)

            if not function_call_matches:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            tool_calls = []
            for i, match in enumerate(function_call_matches):
                function_name, function_args_xml = match
                function_name = function_name.strip()

                # Parse XML arguments to JSON
                function_args_json = self._parse_arguments(function_args_xml)

                tool_calls.append(
                    ToolCall(
                        type='function',
                        function=FunctionCall(name=function_name,
                                              arguments=function_args_json),
                    ))

            # Extract content before the first tool call
            content = model_output[:model_output.find(self.
                                                      tool_calls_start_token)]
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content.strip() if content.strip() else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        self._buffer += delta_text
        cur_text = self._buffer
        start_idx = cur_text.find(self.tool_call_start_token)
        if start_idx == -1:
            self._buffer = ""
            if self.current_tool_id > 0:
                cur_text = ""
            return DeltaMessage(content=cur_text)
        end_idx = cur_text.find(self.tool_call_end_token)
        if end_idx != -1:
            logger.debug("cur_text = %s", cur_text)
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            extracted_tool_calls = self.extract_tool_calls(cur_text[:end_idx + len(self.tool_call_end_token)], request)

            assert len(extracted_tool_calls.tool_calls) == 1

            tool_call = extracted_tool_calls.tool_calls[0]
            self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
            }
            self.streamed_args_for_tool[self.current_tool_id] = tool_call.function.arguments
            delta = DeltaMessage(content=extracted_tool_calls.content, tool_calls=[DeltaToolCall(index=self.current_tool_id, type="function", id=random_tool_call_id(), function=DeltaFunctionCall(name=tool_call.function.name, arguments=tool_call.function.arguments))])
            self.current_tool_id += 1
            self._buffer = cur_text[end_idx + len(self.tool_call_end_token):]
            return delta

        self._buffer = cur_text[start_idx:]
        return DeltaMessage(content=cur_text[:start_idx])


