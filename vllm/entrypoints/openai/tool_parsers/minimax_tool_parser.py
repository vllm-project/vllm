# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Union

import regex as re

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

        # 流式工具调用状态管理
        self.current_tool_name_sent: bool = False
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<tool_calls>"
        self.tool_call_end_token: str = "</tool_calls>"

        self.tool_call_regex = re.compile(
            r"<tool_calls>(.*?)</tool_calls>|<tool_calls>(.*)", re.DOTALL)

        # Add regex pattern for thinking tag
        self.thinking_tag_pattern = r"<think>(.*?)</think>"

        # Simplified buffering for tool calls outside thinking tags
        self.pending_buffer: str = ""
        self.in_thinking_tag: bool = False
        self.thinking_depth: int = 0

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

    def _clean_duplicate_braces(self, args_text: str) -> str:
        import json
        
        args_text = args_text.strip()
        
        if not args_text:
            return args_text
            
        try:
            json.loads(args_text)
            return args_text
        except json.JSONDecodeError:
            pass
        
        while args_text.endswith('}}'):
            candidate = args_text[:-1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                args_text = candidate
        
        return args_text

    def _clean_delta_braces(self, delta_text: str) -> str:
        if not delta_text:
            return delta_text
            
        delta_stripped = delta_text.strip()
        
        if delta_stripped and all(c in '}\n\r\t ' for c in delta_stripped):
            brace_count = delta_stripped.count('}')
            if brace_count > 1:
                if delta_text.endswith('\n'):
                    return '}\n'
                else:
                    return '}'
        
        return delta_text

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

    def _update_thinking_state(self, text: str) -> None:
        """Update the thinking tag state based on current text."""
        open_count = text.count("<think>")
        close_count = text.count("</think>")
        self.in_thinking_tag = open_count > close_count

    def _should_buffer_content(self, delta_text: str) -> bool:
        if self.in_thinking_tag:
            return False

        if self.pending_buffer:
            return True

        start_in_delta_text = self.tool_call_start_token in delta_text
        end_in_delta_text = self.tool_call_end_token in delta_text

        return (start_in_delta_text or end_in_delta_text
                or delta_text.startswith('<'))

    def _split_content_for_buffering(self, delta_text: str) -> tuple[str, str]:
        if self.in_thinking_tag:
            return delta_text, ""

        # Check for potential start of tool call tags
        for tag in [self.tool_call_start_token, self.tool_call_end_token]:
            for i in range(1, len(tag)):
                tag_prefix = tag[:i]
                pos = delta_text.rfind(tag_prefix)
                if pos != -1:
                    # Check if this position could be the start of the tag
                    remaining_text = delta_text[pos:]
                    if tag.startswith(remaining_text):
                        # This could be the start of a tool tag
                        safe_content = delta_text[:pos]
                        potential_tag = delta_text[pos:]
                        return safe_content, potential_tag

        return delta_text, ""

    def _process_buffer(self, new_content: str) -> str:
        """Process the buffer and return content that can be safely output."""
        self.pending_buffer += new_content
        output_content = ""

        # If we're in a thinking tag, output everything as content
        if self.in_thinking_tag:
            output_content = self.pending_buffer
            self.pending_buffer = ""
            return output_content

        # Process the buffer to remove tool call tags
        while self.pending_buffer:
            # Find tool call tags in buffer
            start_pos = self.pending_buffer.find(self.tool_call_start_token)
            end_pos = self.pending_buffer.find(self.tool_call_end_token)

            # Find the first occurring tag (start or end)
            first_tag_pos = -1
            tag_length = 0

            if start_pos != -1 and (end_pos == -1 or start_pos < end_pos):
                first_tag_pos = start_pos
                tag_length = len(self.tool_call_start_token)
            elif end_pos != -1:
                first_tag_pos = end_pos
                tag_length = len(self.tool_call_end_token)

            if first_tag_pos != -1:
                # Output content before the tag
                output_content += self.pending_buffer[:first_tag_pos]
                # Remove content up to and including the tag
                self.pending_buffer = (self.pending_buffer[first_tag_pos +
                                                           tag_length:])
            else:
                potential_tag_start = -1

                # Check for potential start tag
                for i in range(
                        1,
                        min(
                            len(self.pending_buffer) + 1,
                            len(self.tool_call_start_token))):
                    suffix = self.pending_buffer[-i:]
                    if self.tool_call_start_token.startswith(suffix):
                        potential_tag_start = len(self.pending_buffer) - i
                        break

                # Check for potential end tag (might be longer)
                for i in range(
                        1,
                        min(
                            len(self.pending_buffer) + 1,
                            len(self.tool_call_end_token))):
                    suffix = self.pending_buffer[-i:]
                    if self.tool_call_end_token.startswith(suffix):
                        candidate_start = len(self.pending_buffer) - i
                        if (candidate_start < potential_tag_start
                                or potential_tag_start == -1):
                            potential_tag_start = candidate_start

                if potential_tag_start != -1:
                    # Output content before the potential tag
                    output_content += self.pending_buffer[:potential_tag_start]
                    # Keep the potential tag in buffer for next iteration
                    self.pending_buffer = self.pending_buffer[
                        potential_tag_start:]
                    break
                else:
                    # No potential tags, output all buffer content
                    output_content += self.pending_buffer
                    self.pending_buffer = ""
                    break

        return output_content

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

        self._update_thinking_state(current_text)

        if self.in_thinking_tag:
            return DeltaMessage(content=delta_text)

        if self._should_buffer_content(delta_text):
            buffered_output = self._process_buffer(delta_text)
            if buffered_output:
                return DeltaMessage(content=buffered_output)
            else:
                return None

        # Check if we need to split content for partial buffering
        safe_content, potential_tag = self._split_content_for_buffering(
            delta_text)
        if potential_tag:
            # Part of the content needs to be buffered
            self.pending_buffer += potential_tag
            if safe_content:
                return DeltaMessage(content=safe_content)
            else:
                return None

        # Preprocess to remove tool calls from thinking tags
        processed_current_text = self.preprocess_model_output(current_text)

        # If no tool calls detected, return delta as content
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
            tool_start_pos = processed_current_text.find(
                self.tool_call_start_token)
            if tool_start_pos == -1:
                return None

            original_tool_start = current_text.find(self.tool_call_start_token)
            if original_tool_start == -1:
                return None

            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.current_tool_name_sent = False
                self.streamed_args_for_tool = [""]

            tool_content_start = original_tool_start + len(
                self.tool_call_start_token)

            # Check if we can extract function name
            if not self.current_tool_name_sent:
                # Look for complete function name in current text
                name_pattern = r'"name":\s*"([^"]+)"'
                match = re.search(name_pattern,
                                  current_text[tool_content_start:])
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
                args_match = re.search(args_pattern,
                                       current_text[tool_content_start:])
                if args_match:
                    args_start_pos = tool_content_start + args_match.end()

                    # Extract everything after "arguments": as raw text
                    args_text = current_text[args_start_pos:]

                    # Remove trailing </tool_calls> if present
                    end_pos = args_text.find(self.tool_call_end_token)
                    if end_pos != -1:
                        args_text = args_text[:end_pos]
                    
                    # Clean up potential duplicate closing braces
                    args_text = self._clean_duplicate_braces(args_text)

                    # Get what we've already streamed for this tool
                    already_streamed = self.streamed_args_for_tool[
                        self.current_tool_id]

                    if already_streamed:
                        if (args_text != already_streamed
                                and args_text.startswith(already_streamed)):
                            args_delta = extract_intermediate_diff(
                                args_text, already_streamed)
                            # Clean up potential duplicate braces in delta
                            if args_delta:
                                args_delta = self._clean_delta_braces(args_delta)
                                self.streamed_args_for_tool[
                                    self.current_tool_id] = args_text
                                return DeltaMessage(tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(
                                            arguments=args_delta).model_dump(
                                                exclude_none=True))
                                ])
                    else:
                        prev_args_text = ""
                        if previous_text:
                            prev_args_start = previous_text.find(
                                '"arguments":')
                            if prev_args_start != -1:
                                prev_args_match = re.search(
                                    args_pattern,
                                    previous_text[prev_args_start:])
                                if prev_args_match:
                                    prev_args_pos = (prev_args_start +
                                                     prev_args_match.end())
                                    prev_args_text = previous_text[
                                        prev_args_pos:]
                                    # Remove trailing </tool_calls> if present
                                    prev_end_pos = prev_args_text.find(
                                        self.tool_call_end_token)
                                    if prev_end_pos != -1:
                                        prev_args_text = prev_args_text[:
                                                                        prev_end_pos]
                                    
                                    # Clean up potential duplicate closing braces
                                    prev_args_text = self._clean_duplicate_braces(prev_args_text)

                        if prev_args_text and args_text.startswith(
                                prev_args_text) and len(args_text) > len(
                                    prev_args_text):
                            # Calculate the incremental part
                            args_delta = extract_intermediate_diff(
                                args_text, prev_args_text)
                            # Clean up potential duplicate braces in delta
                            if args_delta:
                                args_delta = self._clean_delta_braces(args_delta)
                                self.streamed_args_for_tool[
                                    self.current_tool_id] = args_text
                                return DeltaMessage(tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(
                                            arguments=args_delta).model_dump(
                                                exclude_none=True))
                                ])
                        elif args_text and not prev_args_text:
                            # This is the very first content, send it all
                            # Clean up potential duplicate braces
                            clean_args_text = self._clean_delta_braces(args_text)
                            self.streamed_args_for_tool[
                                self.current_tool_id] = args_text
                            return DeltaMessage(tool_calls=[
                                DeltaToolCall(index=self.current_tool_id,
                                              function=DeltaFunctionCall(
                                                  arguments=clean_args_text).
                                              model_dump(exclude_none=True))
                            ])

            return None

        except Exception:
            logger.exception("An unexpected error occurred during",
                             "streaming tool call handling.")
            return None
