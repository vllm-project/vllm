# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Sequence
from json import JSONDecoder
from typing import Union

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


@ToolParserManager.register_module("phi4_mini_json")
class Phi4MiniJsonToolParser(ToolParser):
    """
    Tool call parser for phi-4-mini models intended for use with the
    examples/tool_chat_template_llama.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser phi4_mini_json 
    are all set
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = [
        ]  # map what has been streamed for each tool so far to a list
        self.bot_token = "functools"

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        print(f"Model output: {model_output}")

        pattern = r'functools\[(.*?)\]'
        matches = re.search(pattern, model_output, re.DOTALL)

        if not matches:
            print("No function calls found")
            return ExtractedToolCallInformation(
                            tools_called=False,
                            tool_calls=[],
                            content=model_output)

    
        try:
            try:
                json_content ='[' + matches.group(1)+ ']'
                function_call_arr = []

                function_call_arr = json.loads(json_content) 
                print(f"Successfully extracted {len(function_call_arr)} function calls")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")

            tool_calls: list[ToolCall] = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(raw_function_call["arguments"] \
                                if "arguments" in raw_function_call \
                                else raw_function_call["parameters"])))
                for raw_function_call in function_call_arr
            ]

            # get any content before  the tool call
            ret = ExtractedToolCallInformation(tools_called=True,
                                               tool_calls=tool_calls,
                                               content=None)
            return ret

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # return information to just treat the tool call as regular JSON
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

        # If not starting with the bot token or doesn't contain function call pattern, treat as regular content
        pattern = r'functools\[(.*?)\]'
        matches = re.search(pattern, current_text, re.DOTALL)

        if not matches:
            return DeltaMessage(content=delta_text)
        
        try:
            try:
                json_content = '[' + matches.group(1) + ']'
                tool_call_arr = []
                is_complete = True
                
                try:
                    tool_call_arr = json.loads(json_content)
                except json.JSONDecodeError:
                    # If we can't parse the complete JSON yet, try partial parsing
                    is_complete = False
                    try:
                        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
                        obj, _ = partial_json_loads(matches.group(1), flags)
                        if isinstance(obj, dict):
                            tool_call_arr = [obj]
                        elif isinstance(obj, list):
                            tool_call_arr = obj
                    except:
                        logger.debug('not enough tokens to parse into JSON yet')
                        return None

            except Exception as e:
                logger.debug(f"Error parsing JSON in streaming: {e}")
                return None

            # If no valid tool calls extracted, return None
            if len(tool_call_arr) == 0:
                return None

            # Normalize the arguments/parameters field
            for tool in tool_call_arr:
                if "parameters" in tool:
                    assert "arguments" not in tool, "model generated both parameters and arguments"
                    tool["arguments"] = tool["parameters"]

            # case: we are starting a new tool in the array
            if len(tool_call_arr) > self.current_tool_id + 1:
                # If we're moving on to a new call, make sure we finish streaming the previous one
                if self.current_tool_id >= 0 and len(self.prev_tool_call_arr) > self.current_tool_id:
                    current_tool_call = self.prev_tool_call_arr[self.current_tool_id]
                    cur_arguments = current_tool_call.get("arguments")
                    function_name = current_tool_call.get("name", "unknown_function")  # Ensure name is not None
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        logger.debug("got arguments diff: %s", argument_diff)
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function", 
                                name=function_name, 
                                arguments=argument_diff
                                ).model_dump(exclude_none=True)
                            )
                        ])
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                    else:
                        delta = None
                else:
                    delta = None
                
                # Set up for the new tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            # Get the current tool call we're working with
            current_tool_call = tool_call_arr[self.current_tool_id] if len(tool_call_arr) > self.current_tool_id else {}


            # If the tool name hasn't been sent yet, send it if available
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                cur_arguments = current_tool_call.get("arguments")
                if function_name:
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",  # Always set type to "function"
                            id=f"chatcmpl-tool-{random_uuid()}",
                            function=DeltaFunctionCall(
                                name=str(function_name),
                                arguments=json.dumps(cur_arguments) if cur_arguments else ""
                            ).model_dump(exclude_none=True)
                        )
                    ])
                    self.current_tool_name_sent = True
                else:
                    delta = None
                # Stream arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                function_name = current_tool_call.get("name", "unknown_function")
                delta = None

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    
                    if is_complete:
                        argument_diff = cur_args_json[sent:]
                    elif self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                        if prev_arguments:
                            prev_args_json = json.dumps(prev_arguments)
                            if cur_args_json != prev_args_json:
                                prefix = find_common_prefix(prev_args_json, cur_args_json)
                                argument_diff = prefix[sent:]
                            else:
                                argument_diff = None
                        else:
                            argument_diff = None
                    else:
                        argument_diff = None

                    if argument_diff is not None:
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",  
                                function=DeltaFunctionCall(
                                    name=function_name,  
                                    arguments=argument_diff
                                ).model_dump(exclude_none=True)
                            )
                        ])
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            
            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return None
