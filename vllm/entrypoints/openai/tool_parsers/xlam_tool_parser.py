# SPDX-License-Identifier: Apache-2.0
import json
import re
from collections.abc import Sequence
from typing import Optional, Union

import partial_json_parser
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.utils import (find_common_prefix,
                                                        is_complete_json,
                                                        partial_json_loads)
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("xlam")
class xLAMToolParser(ToolParser):

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        # State for streaming mode
        self.prev_tool_calls: list[dict] = []
        self.current_tools_sent: list[bool] = []
        self.streamed_args: list[str] = []
        # Regex patterns for preprocessing
        self.json_code_block_patterns = [
            r"```(?:json)?\s*([\s\S]*?)```",
            r"\[TOOL_CALLS\]([\s\S]*?)(?=\n|$)",
            r"<tool_call>([\s\S]*?)</tool_call>",
        ]
        self.thinking_tag_pattern = r"</think>([\s\S]*)"

    def preprocess_model_output(
            self, model_output: str) -> tuple[Optional[str], Optional[str]]:
        """
        Preprocess the model output to extract content and potential tool calls.

        Returns:
            Tuple of (content, potential_tool_calls_json)
        """
        # Check for thinking tag
        thinking_match = re.search(self.thinking_tag_pattern, model_output)
        if thinking_match:
            content = model_output[:thinking_match.start() +
                                   len("</think>")].strip()
            thinking_content = thinking_match.group(1).strip()

            # Try to parse the thinking content as JSON
            try:
                json.loads(thinking_content)
                return content, thinking_content
            except json.JSONDecodeError:
                # If can't parse as JSON, look for JSON code blocks # noqa: E501
                for json_pattern in self.json_code_block_patterns:
                    json_matches = re.findall(json_pattern, thinking_content)
                    if json_matches:
                        for json_str in json_matches:
                            try:
                                json.loads(json_str)
                                return content, json_str
                            except json.JSONDecodeError:
                                continue

        # Check for JSON code blocks in the entire output
        for json_pattern in self.json_code_block_patterns:
            json_matches = re.findall(json_pattern, model_output)
            if json_matches:
                for json_str in json_matches:
                    try:
                        json.loads(json_str)
                        # Extract content by removing the JSON code block
                        content = re.sub(json_pattern, "",
                                         model_output).strip()
                        return content, json_str
                    except json.JSONDecodeError:
                        continue
        # If the entire output is a valid JSON array, treat it as tool calls # noqa: E501
        if model_output.strip().startswith("["):
            try:
                json.loads(model_output)
                return None, model_output
            except json.JSONDecodeError:
                pass

        # If no tool calls found, return the original output as content
        return model_output, None

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        try:
            # Preprocess the model output
            content, potential_tool_calls = self.preprocess_model_output(
                model_output)

            if not potential_tool_calls:
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=content)

            # Parse the potential tool calls as JSON
            tool_calls_data = json.loads(potential_tool_calls)

            # Ensure it's an array
            if not isinstance(tool_calls_data, list):
                logger.debug("Tool calls data is not an array")
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=content or model_output,
                )

            tool_calls: list[ToolCall] = []

            for idx, call in enumerate(tool_calls_data):
                if (not isinstance(call, dict) or "name" not in call
                        or "arguments" not in call):
                    logger.debug("Invalid tool call format at index %d", idx)
                    continue

                tool_call = ToolCall(
                    id=f"call_{idx}_{random_uuid()}",
                    type="function",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=(json.dumps(call["arguments"]) if isinstance(
                            call["arguments"], dict) else call["arguments"]),
                    ),
                )
                tool_calls.append(tool_call)

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content,
            )

        except Exception as e:
            logger.exception("Error extracting tool calls: %s", str(e))
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
        # Preprocess the current text
        _, potential_tool_calls = self.preprocess_model_output(current_text)

        if not potential_tool_calls:
            return DeltaMessage(content=delta_text)

        # Continue with streaming logic on the potential tool calls
        flags = (Allow.ALL if self.current_tool_name_sent else Allow.ALL
                 & ~Allow.STR)  # noqa: E501

        try:
            tool_call_arr = []
            is_complete = []
            try:
                # Parse the JSON array
                start_idx = 0
                while start_idx < len(potential_tool_calls):
                    obj, end_idx = partial_json_loads(
                        potential_tool_calls[start_idx:], flags)
                    is_complete.append(
                        is_complete_json(
                            potential_tool_calls[start_idx:start_idx +
                                                 end_idx]))
                    start_idx += end_idx
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            # Case 1: No tools parsed yet
            if len(tool_call_arr) == 0:
                return None

            # Case 2: Starting a new tool in array
            elif (len(tool_call_arr) > 0
                  and len(tool_call_arr) > self.current_tool_id + 1):

                # Handle any remaining arguments from previous tool
                if self.current_tool_id >= 0:
                    # Get current tool call based on state
                    current_tool_call = tool_call_arr[self.current_tool_id]

                    # Handle the case where current_tool_call might be a list
                    if (isinstance(current_tool_call, dict)
                            and "arguments" in current_tool_call):
                        cur_arguments = current_tool_call.get("arguments")
                        if cur_arguments:
                            cur_args_json = json.dumps(cur_arguments)
                            sent = len(
                                self.streamed_args[self.current_tool_id])
                            argument_diff = cur_args_json[sent:]

                            if argument_diff:
                                delta = DeltaMessage(tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(
                                            arguments=argument_diff).
                                        model_dump(exclude_none=True),
                                    )
                                ])
                                self.streamed_args[
                                    self.current_tool_id] += argument_diff
                                return delta

                # Setup new tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tools_sent.append(False)
                self.streamed_args.append("")
                logger.debug("starting new tool %d", self.current_tool_id)
                return None

            # Get current tool call based on state
            current_tool_call = tool_call_arr[self.current_tool_id]

            # Case 3: Send tool name if not sent yet
            if not self.current_tools_sent[self.current_tool_id]:
                # Handle different types of current_tool_call
                if (isinstance(current_tool_call, dict)
                        and "name" in current_tool_call):
                    function_name = current_tool_call.get("name")
                    if function_name:
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=
                                f"call_{self.current_tool_id}_{random_uuid()}",
                                function=DeltaFunctionCall(
                                    name=function_name).model_dump(
                                        exclude_none=True),
                            )
                        ])
                        self.current_tools_sent[self.current_tool_id] = True
                        return delta
                elif (isinstance(current_tool_call, list)
                      and len(current_tool_call) > 0):
                    # This case should not happen in normal operation, but handle it gracefully  # noqa: E501
                    logger.warning(
                        "Unexpected list in current_tool_call during streaming"
                    )
                return None

            # Case 4: Stream arguments
            else:
                # Handle different types of current_tool_call for arguments
                if (isinstance(current_tool_call, dict)
                        and "arguments" in current_tool_call):
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        sent = len(self.streamed_args[self.current_tool_id])
                        cur_args_json = json.dumps(cur_arguments)
                        prev_arguments = None
                        if (self.prev_tool_calls and len(self.prev_tool_calls)
                                > self.current_tool_id):
                            prev_tool_call = self.prev_tool_calls[
                                self.current_tool_id]
                            if isinstance(prev_tool_call, dict):
                                prev_arguments = prev_tool_call.get(
                                    "arguments")

                        argument_diff = None
                        if is_complete[self.current_tool_id]:
                            argument_diff = cur_args_json[sent:]
                        elif prev_arguments:
                            prev_args_json = json.dumps(prev_arguments)
                            if cur_args_json != prev_args_json:
                                prefix = find_common_prefix(
                                    prev_args_json, cur_args_json)
                                argument_diff = prefix[sent:]

                        if argument_diff is not None:
                            delta = DeltaMessage(tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=argument_diff).model_dump(
                                            exclude_none=True),
                                )
                            ])
                            self.streamed_args[
                                self.current_tool_id] += argument_diff
                            return delta

            self.prev_tool_calls = tool_call_arr
            return None

        except Exception:
            logger.exception("Error in streaming tool calls")
            logger.debug("Skipping chunk due to streaming error")
            return None
