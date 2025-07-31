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
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
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
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)

        # Preprocess to remove tool calls from thinking tags
        processed_current_text = self.preprocess_model_output(current_text)

        if self.tool_call_start_token not in processed_current_text:
            return DeltaMessage(content=delta_text)

        if (self.tool_call_start_token_id is not None
                and self.tool_call_start_token_id in delta_token_ids
                and len(delta_token_ids) == 1):
            return None

        # Handle content before tool calls
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

        try:
            # Check for tool call content at the character level in the entire current text
            tool_start_pos = processed_current_text.find(self.tool_call_start_token)
            if tool_start_pos == -1:
                return None
                
            # Get the tool call content position in original current_text  
            original_tool_start = current_text.find(self.tool_call_start_token)
            if original_tool_start == -1:
                return None
            
            # Initialize tool if this is the first time
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.current_tool_name_sent = False
                self.streamed_args_for_tool = [""]

            # Find what part of the tool call content is new in this delta
            tool_content_start = original_tool_start + len(self.tool_call_start_token)
            
            # Check if we can extract function name
            if not self.current_tool_name_sent:
                # Look for complete function name in current text
                name_pattern = r'"name":\s*"([^"]+)"'
                import re
                match = re.search(name_pattern, current_text[tool_content_start:])
                if match:
                    function_name = match.group(1)
                    self.current_tool_name_sent = True
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      type="function",
                                      id=random_tool_call_id(),
                                      function=DeltaFunctionCall(
                                          name=function_name).model_dump(
                                              exclude_none=True))
                    ])

            # Check for arguments content
            if self.current_tool_name_sent:
                # Find arguments start position
                args_pattern = r'"arguments":\s*'
                import re
                args_match = re.search(args_pattern, current_text[tool_content_start:])
                if args_match:
                    args_start_pos = tool_content_start + args_match.end()
                    
                    # Extract everything after "arguments": as raw text
                    args_text = current_text[args_start_pos:]
                    
                    # Remove trailing </tool_calls> if present
                    end_pos = args_text.find(self.tool_call_end_token)
                    if end_pos != -1:
                        args_text = args_text[:end_pos]
                    
                    # Get what we've already streamed for this tool
                    already_streamed = self.streamed_args_for_tool[self.current_tool_id]
                    
                    if already_streamed:
                        # We have previous content, compute the diff
                        if args_text != already_streamed and args_text.startswith(already_streamed):
                            # Use extract_intermediate_diff to get only the new part
                            args_delta = extract_intermediate_diff(args_text, already_streamed)
                            if args_delta:
                                self.streamed_args_for_tool[self.current_tool_id] = args_text
                                return DeltaMessage(tool_calls=[
                                    DeltaToolCall(index=self.current_tool_id,
                                                  function=DeltaFunctionCall(
                                                      arguments=args_delta).model_dump(
                                                          exclude_none=True))
                                ])
                    else:
                        # First time sending arguments
                        # We need to check what was in the previous_text to determine the actual delta
                        prev_args_text = ""
                        if previous_text:
                            prev_args_start = previous_text.find('"arguments":')
                            if prev_args_start != -1:
                                prev_args_match = re.search(args_pattern, previous_text[prev_args_start:])
                                if prev_args_match:
                                    prev_args_pos = prev_args_start + prev_args_match.end()
                                    prev_args_text = previous_text[prev_args_pos:]
                                    # Remove trailing </tool_calls> if present
                                    prev_end_pos = prev_args_text.find(self.tool_call_end_token)
                                    if prev_end_pos != -1:
                                        prev_args_text = prev_args_text[:prev_end_pos]
                        
                        if prev_args_text and args_text.startswith(prev_args_text) and len(args_text) > len(prev_args_text):
                            # Calculate the incremental part
                            args_delta = extract_intermediate_diff(args_text, prev_args_text)
                            if args_delta:
                                self.streamed_args_for_tool[self.current_tool_id] = args_text
                                return DeltaMessage(tool_calls=[
                                    DeltaToolCall(index=self.current_tool_id,
                                                  function=DeltaFunctionCall(
                                                      arguments=args_delta).model_dump(
                                                          exclude_none=True))
                                ])
                        elif args_text and not prev_args_text:
                            # This is the very first content, send it all
                            self.streamed_args_for_tool[self.current_tool_id] = args_text
                            return DeltaMessage(tool_calls=[
                                DeltaToolCall(index=self.current_tool_id,
                                              function=DeltaFunctionCall(
                                                  arguments=args_text).model_dump(
                                                      exclude_none=True))
                            ])

            return None

        except Exception:
            logger.exception("An unexpected error occurred during streaming tool call handling.")
            return None