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

        self.tool_states: list[dict] = []
        self.current_tool_index: int = -1
        
        self.tool_call_start_token: str = "<tool_calls>"
        self.tool_call_end_token: str = "</tool_calls>"

        self.tool_call_regex = re.compile(
            r"<tool_calls>(.*?)</tool_calls>|<tool_calls>(.*)", re.DOTALL)

        self.thinking_tag_pattern = r"<think>(.*?)</think>"

        self.tool_name_pattern = re.compile(r'"name":\s*"([^"]+)"')
        self.tool_args_pattern = re.compile(r'"arguments":\s*')

        self.pending_buffer: str = ""
        self.in_thinking_tag: bool = False

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
        def remove_tool_calls_from_think(match):
            think_content = match.group(1)
            cleaned_content = re.sub(r"<tool_calls>.*?</tool_calls>",
                                     "",
                                     think_content,
                                     flags=re.DOTALL)
            return f"<think>{cleaned_content}</think>"

        processed_output = re.sub(self.thinking_tag_pattern,
                                  remove_tool_calls_from_think,
                                  model_output,
                                  flags=re.DOTALL)
        return processed_output

    def _clean_duplicate_braces(self, args_text: str) -> str:
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
                        if line and line.startswith('{') and line.endswith('}'):
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

            processed_pos = processed_output.find(self.tool_call_start_token)
            if processed_pos != -1:
                processed_content = processed_output[:processed_pos].strip()

                if processed_content:
                    lines = processed_content.split('\n')
                    for line in reversed(lines):
                        line = line.strip()
                        if line:
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

        for tag in [self.tool_call_start_token, self.tool_call_end_token]:
            for i in range(1, len(tag)):
                tag_prefix = tag[:i]
                pos = delta_text.rfind(tag_prefix)
                if pos != -1:
                    remaining_text = delta_text[pos:]
                    if tag.startswith(remaining_text):
                        safe_content = delta_text[:pos]
                        potential_tag = delta_text[pos:]
                        return safe_content, potential_tag

        return delta_text, ""

    def _process_buffer(self, new_content: str) -> str:
        self.pending_buffer += new_content
        output_content = ""

        if self.in_thinking_tag:
            output_content = self.pending_buffer
            self.pending_buffer = ""
            return output_content

        while self.pending_buffer:
            start_pos = self.pending_buffer.find(self.tool_call_start_token)
            end_pos = self.pending_buffer.find(self.tool_call_end_token)

            first_tag_pos = -1
            tag_length = 0

            if start_pos != -1 and (end_pos == -1 or start_pos < end_pos):
                first_tag_pos = start_pos
                tag_length = len(self.tool_call_start_token)
            elif end_pos != -1:
                first_tag_pos = end_pos
                tag_length = len(self.tool_call_end_token)

            if first_tag_pos != -1:
                output_content += self.pending_buffer[:first_tag_pos]
                self.pending_buffer = (self.pending_buffer[first_tag_pos +
                                                           tag_length:])
            else:
                potential_tag_start = -1

                for i in range(
                        1,
                        min(
                            len(self.pending_buffer) + 1,
                            len(self.tool_call_start_token))):
                    suffix = self.pending_buffer[-i:]
                    if self.tool_call_start_token.startswith(suffix):
                        potential_tag_start = len(self.pending_buffer) - i
                        break

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
                    output_content += self.pending_buffer[:potential_tag_start]
                    self.pending_buffer = self.pending_buffer[
                        potential_tag_start:]
                    break
                else:
                    output_content += self.pending_buffer
                    self.pending_buffer = ""
                    break

        return output_content

    def _reset_tool_states(self) -> None:
        self.tool_states = []
        self.current_tool_index = -1

    def _ensure_tool_state(self, tool_index: int) -> None:
        while len(self.tool_states) <= tool_index:
            self.tool_states.append({
                'id': random_tool_call_id(),
                'name_sent': False,
                'name': None,
                'args_sent': ''
            })

    def _detect_tools_in_text(self, text: str) -> int:
        matches = self.tool_name_pattern.findall(text)
        return len(matches)

    def _find_tool_boundaries(self, text: str) -> list[tuple[int, int]]:
        boundaries = []
        
        i = 0
        while i < len(text):
            if text[i] == '{':
                start = i
                depth = 0
                while i < len(text):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            segment = text[start:end]
                            if '"name"' in segment and '"arguments"' in segment:
                                boundaries.append((start, end))
                            break
                    i += 1
            else:
                i += 1
        
        return boundaries

    def _get_current_tool_content(self, text: str, tool_index: int) -> tuple[str, str]:
        boundaries = self._find_tool_boundaries(text)
        
        if tool_index >= len(boundaries):
            return None, None
            
        start, end = boundaries[tool_index]
        tool_content = text[start:end]
        
        name_match = self.tool_name_pattern.search(tool_content)
        name = name_match.group(1) if name_match else None
        
        args_match = self.tool_args_pattern.search(tool_content)
        if args_match:
            args_start_pos = args_match.end()
            remaining_content = tool_content[args_start_pos:]
            
            try:
                if remaining_content.strip().startswith('{'):
                    depth = 0
                    args_end = -1
                    for i, char in enumerate(remaining_content):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                args_end = i + 1
                                break
                    
                    if args_end > 0:
                        args_text = remaining_content[:args_end]
                        return name, args_text
                else:
                    args_end = remaining_content.find('}')
                    if args_end > 0:
                        args_text = remaining_content[:args_end].strip()
                        return name, args_text
            except Exception:
                args_text = remaining_content.rstrip('}').strip()
                return name, args_text
        
        return name, None

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

        safe_content, potential_tag = self._split_content_for_buffering(
            delta_text)
        if potential_tag:
            self.pending_buffer += potential_tag
            if safe_content:
                return DeltaMessage(content=safe_content)
            else:
                return None

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

        try:
            tool_start_pos = processed_current_text.find(
                self.tool_call_start_token)
            if tool_start_pos == -1:
                return None

            original_tool_start = current_text.find(self.tool_call_start_token)
            if original_tool_start == -1:
                return None

            tool_content_start = original_tool_start + len(self.tool_call_start_token)
            tool_content = current_text[tool_content_start:]
            
            end_pos = tool_content.find(self.tool_call_end_token)
            if end_pos != -1:
                tool_content = tool_content[:end_pos]

            current_tools_count = self._detect_tools_in_text(tool_content)
            
            if current_tools_count == 0:
                return None

            if self.current_tool_index == -1:
                self._reset_tool_states()
                self.current_tool_index = 0

            for tool_idx in range(current_tools_count):
                self._ensure_tool_state(tool_idx)
                
                tool_name, tool_args = self._get_current_tool_content(tool_content, tool_idx)
                
                if not tool_name:
                    continue
                    
                tool_state = self.tool_states[tool_idx]
                
                if not tool_state['name_sent'] and tool_idx <= self.current_tool_index + 1:
                    tool_state['name'] = tool_name
                    tool_state['name_sent'] = True
                    self.current_tool_index = tool_idx
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=tool_idx,
                            type="function",
                            id=tool_state['id'],
                            function=DeltaFunctionCall(name=tool_name).model_dump(
                                exclude_none=True))
                    ])
                
                if (tool_state['name_sent'] and tool_args is not None and 
                    tool_idx <= self.current_tool_index + 1):
                    clean_args = self._clean_duplicate_braces(tool_args)
                    
                    if clean_args != tool_state['args_sent']:
                        if tool_state['args_sent'] and clean_args.startswith(tool_state['args_sent']):
                            args_delta = extract_intermediate_diff(
                                clean_args, tool_state['args_sent'])
                            
                            if args_delta:
                                args_delta = self._clean_delta_braces(args_delta)
                                tool_state['args_sent'] = clean_args
                                self.current_tool_index = tool_idx
                                
                                return DeltaMessage(tool_calls=[
                                    DeltaToolCall(
                                        index=tool_idx,
                                        function=DeltaFunctionCall(
                                            arguments=args_delta).model_dump(
                                                exclude_none=True))
                                ])
                        elif not tool_state['args_sent'] and clean_args:
                            clean_args_delta = self._clean_delta_braces(clean_args)
                            tool_state['args_sent'] = clean_args
                            self.current_tool_index = tool_idx
                            
                            return DeltaMessage(tool_calls=[
                                DeltaToolCall(
                                    index=tool_idx,
                                    function=DeltaFunctionCall(
                                        arguments=clean_args_delta).model_dump(
                                            exclude_none=True))
                            ])

            return None

        except Exception:
            logger.exception("An unexpected error occurred during "
                             "streaming tool call handling.")
            return None