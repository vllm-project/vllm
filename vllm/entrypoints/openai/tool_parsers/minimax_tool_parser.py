# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Union

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@ToolParserManager.register_module("minimax")
class MinimaxToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = [
        ]  # map what has been streamed for each tool so far to a list

        # Minimax uses <tool_calls> tags instead of <tool_call>
        self.tool_call_start_token: str = "<tool_calls>"
        self.tool_call_end_token: str = "</tool_calls>"

        # Regex pattern for Minimax tool calls format
        # Matches both complete and incomplete tool calls
        self.tool_call_regex = re.compile(
            r"<tool_calls>(.*?)</tool_calls>|<tool_calls>(.*)", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")
        
        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        
        if (self.tool_call_start_token_id is None
                or self.tool_call_end_token_id is None):
            logger.warning(
                "Minimax Tool parser could not locate tool call start/end "
                "tokens in the tokenizer. Falling back to string matching.")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        # sanity check; avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            # Extract tool calls content between tags
            function_call_tuples = (
                self.tool_call_regex.findall(model_output))

            # Parse each tool call JSON
            raw_function_calls = []
            for match in function_call_tuples:
                tool_call_content = match[0] if match[0] else match[1]
                if tool_call_content.strip():
                    # Split by lines and parse each JSON object
                    lines = tool_call_content.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and line.startswith('{') and line.endswith('}'):
                            try:
                                parsed_call = json.loads(line)
                                raw_function_calls.append(parsed_call)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse tool call JSON: {line}")
                                continue

            # Convert to ToolCall objects
            tool_calls = []
            for function_call in raw_function_calls:
                if "name" in function_call and "arguments" in function_call:
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=function_call["name"],
                                # function call args are JSON but as a string
                                arguments=json.dumps(function_call["arguments"],
                                                   ensure_ascii=False)))
                    )

            # Extract content before tool calls
            content = model_output[:model_output.find(self.tool_call_start_token)]
            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content.strip() if content.strip() else None)

        except Exception:
            logger.exception(
                "Error in extracting tool call from response.")
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

        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        
        # Check if we have tool call tokens or string patterns
        has_tool_call_tokens = (self.tool_call_start_token_id is not None and 
                               self.tool_call_start_token_id in current_token_ids)
        has_tool_call_string = self.tool_call_start_token in current_text
        
        if not (has_tool_call_tokens or has_tool_call_string):
            logger.debug("No tool call tokens or strings found!")
            return DeltaMessage(content=delta_text)

        try:
            # Count tool call start & end tags for state tracking
            prev_tool_start_count = previous_text.count(self.tool_call_start_token)
            prev_tool_end_count = previous_text.count(self.tool_call_end_token)
            cur_tool_start_count = current_text.count(self.tool_call_start_token)
            cur_tool_end_count = current_text.count(self.tool_call_end_token)
            
            tool_call_portion = None
            text_portion = None

            # Case: generating regular text content
            if (cur_tool_start_count == cur_tool_end_count
                    and prev_tool_end_count == cur_tool_end_count
                    and self.tool_call_end_token not in delta_text):
                logger.debug("Generating text content! skipping tool parsing.")
                return DeltaMessage(content=delta_text)

            # Case: tool call is being closed
            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text
                if self.tool_call_start_token in full_text:
                    tool_call_portion = full_text.split(
                        self.tool_call_start_token)[-1].split(
                            self.tool_call_end_token)[0].strip()
                text_portion = delta_text.split(
                    self.tool_call_end_token)[-1].strip()

            # Flags for partial JSON parsing
            flags = Allow.ALL if self.current_tool_name_sent \
                else Allow.ALL & ~Allow.STR

            # Case: starting a new tool call
            if (cur_tool_start_count > cur_tool_end_count
                    and cur_tool_start_count > prev_tool_start_count):
                
                if self.tool_call_start_token in current_text:
                    tool_call_portion = current_text.split(
                        self.tool_call_start_token)[-1]
                else:
                    tool_call_portion = None

                text_portion = None

                # Set cursors and state appropriately
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # Case: updating an existing tool call
            elif (cur_tool_start_count > cur_tool_end_count
                  and cur_tool_start_count == prev_tool_start_count):

                if self.tool_call_start_token in current_text:
                    tool_call_portion = current_text.split(
                        self.tool_call_start_token)[-1]
                text_portion = None

            # Case: tool call is being closed
            elif (cur_tool_start_count == cur_tool_end_count
                  and cur_tool_end_count > prev_tool_end_count):
                
                if (self.prev_tool_call_arr is None
                        or len(self.prev_tool_call_arr) == 0):
                    logger.debug(
                        "attempting to close tool call, but no tool call")
                    return None
                
                # Handle final arguments if any
                if self.current_tool_id >= 0 and self.current_tool_id < len(self.prev_tool_call_arr):
                    diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                    if diff and '"}' in delta_text:
                        end_loc = delta_text.rindex('"}')
                        diff = delta_text[:end_loc] + '"}'
                        logger.debug(
                            "Finishing tool and found diff that had not "
                            "been streamed yet: %s", diff)
                        self.streamed_args_for_tool[self.current_tool_id] += diff
                        return DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id,
                                          function=DeltaFunctionCall(
                                              arguments=diff).model_dump(
                                                  exclude_none=True))
                        ])

            # Case: generating regular text
            else:
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                return DeltaMessage(content=text)

            # Parse the current tool call portion
            try:
                current_tool_call = None
                if tool_call_portion:
                    # Try to parse as complete JSON first
                    lines = tool_call_portion.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith('{') or '"name"' in line):
                            try:
                                current_tool_call = partial_json_parser.loads(
                                    line, flags)
                                break
                            except:
                                continue
                    
                    if not current_tool_call:
                        current_tool_call = partial_json_parser.loads(
                            tool_call_portion, flags)
                
                logger.debug("Parsed tool call %s", current_tool_call)
            except (partial_json_parser.core.exceptions.MalformedJSON, 
                    json.decoder.JSONDecodeError):
                logger.debug('not enough tokens to parse into JSON yet')
                return None

            # Case: haven't sent the tool name yet
            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name: Union[str, None] = current_tool_call.get("name")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      type="function",
                                      id=random_tool_call_id(),
                                      function=DeltaFunctionCall(
                                          name=function_name).model_dump(
                                              exclude_none=True))
                    ])
                else:
                    return None

            # Handle tool call portion
            if tool_call_portion is None:
                delta = DeltaMessage(content=delta_text) \
                    if text_portion is not None else None
                return delta

            logger.debug("Trying to parse current tool call with ID %s",
                         self.current_tool_id)

            # Initialize tool call array if needed
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # Compare previous and current arguments
            prev_arguments = (
                self.prev_tool_call_arr[self.current_tool_id].get("arguments"))
            cur_arguments = current_tool_call.get("arguments") if current_tool_call else None

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            # Handle different argument states
            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None
            elif not cur_arguments and prev_arguments:
                logger.error("should be impossible to have arguments reset "
                             "mid-call. skipping streaming anything.")
                delta = None
            elif cur_arguments and not prev_arguments:
                cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
                logger.debug("finding %s in %s", delta_text, cur_arguments_json)

                if delta_text not in cur_arguments_json[:-2]:
                    return None
                
                args_delta_start_loc = cur_arguments_json[:-2].rindex(delta_text) + len(delta_text)
                arguments_delta = cur_arguments_json[:args_delta_start_loc]
                logger.debug("First tokens in arguments received: %s", arguments_delta)

                delta = DeltaMessage(tool_calls=[
                    DeltaToolCall(index=self.current_tool_id,
                                  function=DeltaFunctionCall(
                                      arguments=arguments_delta).model_dump(
                                          exclude_none=True))
                ])
                self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
            elif cur_arguments and prev_arguments:
                if isinstance(delta_text, str) and len(delta_text.strip()) >= 1 and delta_text.strip()[-1] == '}':
                    delta_text = delta_text.strip()[:-1]

                logger.debug("got diff %s", delta_text)

                delta = DeltaMessage(tool_calls=[
                    DeltaToolCall(index=self.current_tool_id,
                                  function=DeltaFunctionCall(
                                      arguments=delta_text).model_dump(
                                          exclude_none=True))
                ])
                self.streamed_args_for_tool[self.current_tool_id] += delta_text

            # Save state for next iteration
            if current_tool_call:
                if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                    self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
                else:
                    self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID. 