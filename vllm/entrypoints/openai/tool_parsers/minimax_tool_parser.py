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
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_calls>"
        self.tool_call_end_token: str = "</tool_calls>"

        self.tool_call_regex = re.compile(
            r"<tool_calls>(.*?)</tool_calls>|<tool_calls>(.*)", re.DOTALL)

        # Add regex pattern for thinking tag
        self.thinking_tag_pattern = r"<think>(.*?)</think>"

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

    def preprocess_model_output(self, model_output: str) -> str:
        """
        Remove tool calls from within thinking tags to avoid processing them.
        """

        def remove_tool_calls_from_think(match):
            think_content = match.group(1)
            # Remove tool_calls from within the think tag
            cleaned_content = re.sub(r"<tool_calls>.*?</tool_calls>",
                                     "",
                                     think_content,
                                     flags=re.DOTALL)
            return f"<think>{cleaned_content}</think>"

        # Process thinking tags and remove tool_calls from within them
        processed_output = re.sub(self.thinking_tag_pattern,
                                  remove_tool_calls_from_think,
                                  model_output,
                                  flags=re.DOTALL)

        return processed_output

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        # Preprocess to remove tool calls from thinking tags
        processed_output = self.preprocess_model_output(model_output)

        if self.tool_call_start_token not in processed_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            function_call_tuples = (
                self.tool_call_regex.findall(processed_output))

            raw_function_calls = []
            for match in function_call_tuples:
                tool_call_content = match[0] if match[0] else match[1]
                if tool_call_content.strip():
                    lines = tool_call_content.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and line.startswith('{') and line.endswith(
                                '}'):
                            try:
                                parsed_call = json.loads(line)
                                raw_function_calls.append(parsed_call)
                            except json.JSONDecodeError:
                                continue

            tool_calls = []
            for function_call in raw_function_calls:
                if "name" in function_call and "arguments" in function_call:
                    tool_calls.append(
                        ToolCall(type="function",
                                 function=FunctionCall(
                                     name=function_call["name"],
                                     arguments=json.dumps(
                                         function_call["arguments"],
                                         ensure_ascii=False))))

            # Extract content before the first valid tool call
            # Find the position in processed output, then map back to original
            processed_pos = processed_output.find(self.tool_call_start_token)
            if processed_pos != -1:
                # Get the content before tool calls in processed output
                processed_content = processed_output[:processed_pos].strip()

                if processed_content:
                    # Find the end of this content in the original output
                    # Look for the last non-empty line of processed content
                    lines = processed_content.split('\n')
                    for line in reversed(lines):
                        line = line.strip()
                        if line:
                            # Find this line in original output
                            pos = model_output.find(line)
                            if pos != -1:
                                content = model_output[:pos + len(line)]
                                break
                    else:
                        content = ""
                else:
                    content = ""
            else:
                content = model_output

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content.strip() if content.strip() else None)

        except Exception:
            logger.exception(
                "An unexpected error occurred during tool call extraction.")
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


        # Preprocess to remove tool calls from thinking tags
        processed_current_text = self.preprocess_model_output(current_text)

        if self.tool_call_start_token not in processed_current_text:
            return DeltaMessage(content=delta_text)

        if (self.tool_call_start_token_id is not None
                and self.tool_call_start_token_id in delta_token_ids
                and len(delta_token_ids) == 1):
            return None

        original_tool_call_start_pos = current_text.find(
            self.tool_call_start_token)
        if original_tool_call_start_pos > 0:
            delta_start_pos = len(current_text) - len(delta_text)
            if delta_start_pos < original_tool_call_start_pos:
                content_part = delta_text
                if delta_start_pos + len(
                        delta_text) > original_tool_call_start_pos:
                    content_part = delta_text[:original_tool_call_start_pos -
                                              delta_start_pos]
                if content_part:
                    return DeltaMessage(content=content_part)

        flags = Allow.ALL if self.current_tool_name_sent \
            else Allow.ALL & ~Allow.STR

        try:
            parsable_content = processed_current_text.split(
                self.tool_call_start_token)[-1].split(
                    self.tool_call_end_token)[0]

            # Extract previous parsable content for delta calculation
            previous_parsable_content = ""
            if previous_text:
                previous_processed = self.preprocess_model_output(previous_text)
                if self.tool_call_start_token in previous_processed:
                    previous_parsable_content = previous_processed.split(
                        self.tool_call_start_token)[-1].split(
                            self.tool_call_end_token)[0]

            tool_call_arr = []
            if parsable_content.strip():
                lines = parsable_content.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('{') or '"name"' in line):
                        try:
                            if line.endswith('}'):
                                parsed_call = json.loads(line)
                                tool_call_arr.append(parsed_call)
                            else:
                                parsed_call = partial_json_parser.loads(
                                    line, flags)
                                if parsed_call and isinstance(
                                        parsed_call, dict):
                                    tool_call_arr.append(parsed_call)
                        except (json.JSONDecodeError, partial_json_parser.core.
                                exceptions.MalformedJSON):
                            continue

            current_tool_call: dict = tool_call_arr[self.current_tool_id] \
                if len(tool_call_arr) > self.current_tool_id >= 0 else {}

            # Detect new tools at string level, not just relying on JSON parsing
            current_content_lines = parsable_content.strip().split('\n') if parsable_content.strip() else []
            
            # Initialize tool ID if not set but content lines exist
            if self.current_tool_id == -1 and len(current_content_lines) > 0:
                self.current_tool_id = 0
                self.current_tool_name_sent = False
                self.streamed_args_for_tool = [""]

            if len(tool_call_arr) == 0:
                # Continue processing even if JSON parsing fails
                pass

            # Starting a new tool in the array
            elif (len(tool_call_arr) > 0
                  and len(tool_call_arr) > self.current_tool_id + 1):

                # Handle any missed arguments from previous tool
                if self.current_tool_id >= 0 and self.current_tool_id < len(
                        self.prev_tool_call_arr):
                    prev_tool_call = self.prev_tool_call_arr[
                        self.current_tool_id]
                    diff_arguments = prev_tool_call.get("arguments")

                    if diff_arguments:
                        diff_arguments_json = json.dumps(diff_arguments,
                                                         ensure_ascii=False)
                        already_streamed = self.streamed_args_for_tool[
                            self.
                            current_tool_id] if self.current_tool_id < len(
                                self.streamed_args_for_tool) else ""

                        if diff_arguments_json != already_streamed:
                            diff = diff_arguments_json[len(already_streamed):]
                            delta = DeltaMessage(tool_calls=[
                                DeltaToolCall(index=self.current_tool_id,
                                              function=DeltaFunctionCall(
                                                  arguments=diff).model_dump(
                                                      exclude_none=True))
                            ])
                            if self.current_tool_id < len(
                                    self.streamed_args_for_tool):
                                self.streamed_args_for_tool[
                                    self.current_tool_id] = diff_arguments_json
                        else:
                            delta = None
                    else:
                        delta = None
                else:
                    delta = None

                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                return delta

            # Send tool name if not sent yet
            function_name_delta = None
            if not self.current_tool_name_sent and self.current_tool_id >= 0:
                function_name = current_tool_call.get("name")
                # Extract function name from string if JSON parsing failed
                if not function_name and len(current_content_lines) > self.current_tool_id:
                    tool_line = current_content_lines[self.current_tool_id].strip()
                    if '"name":' in tool_line:
                        import re
                        name_match = re.search(r'"name":\s*"([^"]+)"', tool_line)
                        if name_match:
                            function_name = name_match.group(1)
                
                if function_name:
                    function_name_delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      type="function",
                                      id=random_tool_call_id(),
                                      function=DeltaFunctionCall(
                                          name=function_name).model_dump(
                                              exclude_none=True))
                    ])
                    self.current_tool_name_sent = True

            # Stream arguments - process at string level, independent of JSON parsing
            if True:  # Always check for argument changes
                # Compare parsable content differences
                previous_content_lines = previous_parsable_content.strip().split('\n') if previous_parsable_content.strip() else []
                
                # Determine current tool line index
                tool_line_idx = max(0, self.current_tool_id) if self.current_tool_id >= 0 else 0
                
                # Get current and previous tool lines
                current_tool_line = ""
                previous_tool_line = ""
                
                if tool_line_idx < len(current_content_lines):
                    current_tool_line = current_content_lines[tool_line_idx].strip()
                
                if tool_line_idx < len(previous_content_lines):
                    previous_tool_line = previous_content_lines[tool_line_idx].strip()
                
                # Calculate delta for this line at string level
                if current_tool_line and current_tool_line != previous_tool_line:
                    # Find arguments value start position
                    if '"arguments":' in current_tool_line:
                        args_key_pos = current_tool_line.find('"arguments":')
                        args_value_start = args_key_pos + len('"arguments":')
                        while args_value_start < len(current_tool_line) and current_tool_line[args_value_start] in ' \t':
                            args_value_start += 1
                        current_args_value = current_tool_line[args_value_start:]
                        
                        # Get previous arguments value, preferring recorded state
                        previous_args_value = ""
                        tool_id = max(0, self.current_tool_id)
                        if tool_id < len(self.streamed_args_for_tool):
                            previous_args_value = self.streamed_args_for_tool[tool_id]
                        
                        # Fallback to extracting from previous_tool_line if no recorded state
                        if not previous_args_value and previous_tool_line and '"arguments":' in previous_tool_line:
                            prev_args_key_pos = previous_tool_line.find('"arguments":')
                            prev_args_value_start = prev_args_key_pos + len('"arguments":')
                            while prev_args_value_start < len(previous_tool_line) and previous_tool_line[prev_args_value_start] in ' \t':
                                prev_args_value_start += 1
                            previous_args_value = previous_tool_line[prev_args_value_start:]
                        
                        # Calculate delta: current args minus previous args
                        # Handle JSON closing braces for proper prefix matching
                        previous_args_clean = previous_args_value
                        if previous_args_value and previous_args_value.endswith('}}'):
                            previous_args_clean = previous_args_value[:-2]
                        elif previous_args_value and previous_args_value.endswith('}'):
                            previous_args_clean = previous_args_value[:-1]
                            
                        if current_args_value and previous_args_clean and current_args_value.startswith(previous_args_clean):
                            argument_diff = current_args_value[len(previous_args_clean):]
                        elif current_args_value and current_args_value != previous_args_value:
                            argument_diff = current_args_value
                        else:
                            argument_diff = ""
                        
                        # Update streamed args state regardless of whether there's a delta
                        tool_id = max(0, self.current_tool_id)
                        while len(self.streamed_args_for_tool) <= tool_id:
                            self.streamed_args_for_tool.append("")
                        self.streamed_args_for_tool[tool_id] = current_args_value
                        
                        if argument_diff:
                            arguments_delta = DeltaMessage(tool_calls=[
                                DeltaToolCall(index=max(0, self.current_tool_id),
                                              function=DeltaFunctionCall(
                                                  arguments=argument_diff).
                                              model_dump(exclude_none=True))
                            ])
                        else:
                            arguments_delta = None
                    else:
                        arguments_delta = None
                else:
                    arguments_delta = None

            self.prev_tool_call_arr = tool_call_arr
            
            # Merge function name and arguments deltas, prioritize function name (OpenAI behavior)
            if function_name_delta is not None:
                return function_name_delta
            elif arguments_delta is not None:
                return arguments_delta
            else:
                return None

        except Exception:
            logger.exception("An unexpected error occurred",
                             "during streaming tool call handling.")
            return None
