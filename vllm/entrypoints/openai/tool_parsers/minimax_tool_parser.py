# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
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
)
from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


class MinimaxToolParser(ToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # Initialize streaming state for tracking tool call progress
        self.streaming_state: dict[str, Any] = {
            "current_tool_index": -1,  # Index of current tool being processed
            "tool_ids": [],  # List of tool call IDs
            "sent_tools": [],  # List of tools that have been sent
        }

        # Define tool call tokens and patterns
        self.tool_call_start_token = "<tool_calls>"
        self.tool_call_end_token = "</tool_calls>"
        self.tool_call_regex = re.compile(
            r"<tool_calls>(.*?)</tool_calls>|<tool_calls>(.*)", re.DOTALL
        )
        self.thinking_tag_pattern = r"<think>(.*?)</think>"
        self.tool_name_pattern = re.compile(r'"name":\s*"([^"]+)"')
        self.tool_args_pattern = re.compile(r'"arguments":\s*')

        # Buffer for handling partial tool calls during streaming
        self.pending_buffer = ""
        self.in_thinking_tag = False

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Get token IDs for tool call start/end tokens
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            logger.warning(
                "Minimax Tool parser could not locate tool call start/end "
                "tokens in the tokenizer. Falling back to string matching."
            )

    def preprocess_model_output(self, model_output: str) -> str:
        """
        Preprocess model output by removing tool calls from thinking tags.

        Args:
            model_output: Raw model output string

        Returns:
            Preprocessed model output with tool calls removed from thinking tags
        """

        def remove_tool_calls_from_think(match):
            think_content = match.group(1)
            cleaned_content = re.sub(
                r"<tool_calls>.*?</tool_calls>", "", think_content, flags=re.DOTALL
            )
            return f"<think>{cleaned_content}</think>"

        return re.sub(
            self.thinking_tag_pattern,
            remove_tool_calls_from_think,
            model_output,
            flags=re.DOTALL,
        )

    def _clean_duplicate_braces(self, args_text: str) -> str:
        """
        Clean duplicate closing braces from arguments text.

        Args:
            args_text: Raw arguments text

        Returns:
            Cleaned arguments text with proper JSON formatting
        """
        args_text = args_text.strip()
        if not args_text:
            return args_text

        try:
            json.loads(args_text)
            return args_text
        except json.JSONDecodeError:
            pass

        while args_text.endswith("}}"):
            candidate = args_text[:-1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                args_text = candidate

        return args_text

    def _clean_delta_braces(self, delta_text: str) -> str:
        """
        Clean delta text by removing excessive closing braces.

        Args:
            delta_text: Delta text to clean

        Returns:
            Cleaned delta text
        """
        if not delta_text:
            return delta_text

        delta_stripped = delta_text.strip()

        if delta_stripped and all(c in "}\n\r\t " for c in delta_stripped):
            brace_count = delta_stripped.count("}")
            if brace_count > 1:
                return "}\n" if delta_text.endswith("\n") else "}"

        return delta_text

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from model output for non-streaming mode.

        Args:
            model_output: Complete model output
            request: Chat completion request

        Returns:
            ExtractedToolCallInformation containing tool calls and content
        """
        processed_output = self.preprocess_model_output(model_output)

        if self.tool_call_start_token not in processed_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_call_tuples = self.tool_call_regex.findall(processed_output)

            raw_function_calls = []
            for match in function_call_tuples:
                tool_call_content = match[0] if match[0] else match[1]
                if tool_call_content.strip():
                    lines = tool_call_content.strip().split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and line.startswith("{") and line.endswith("}"):
                            try:
                                parsed_call = json.loads(line)
                                raw_function_calls.append(parsed_call)
                            except json.JSONDecodeError:
                                continue

            tool_calls = []
            for function_call in raw_function_calls:
                if "name" in function_call and "arguments" in function_call:
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=function_call["name"],
                                arguments=json.dumps(
                                    function_call["arguments"], ensure_ascii=False
                                ),
                            ),
                        )
                    )

            processed_pos = processed_output.find(self.tool_call_start_token)
            if processed_pos != -1:
                processed_content = processed_output[:processed_pos].strip()

                if processed_content:
                    lines = processed_content.split("\n")
                    for line in reversed(lines):
                        line = line.strip()
                        if line:
                            pos = model_output.find(line)
                            if pos != -1:
                                content = model_output[: pos + len(line)]
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
                content=content.strip() if content.strip() else None,
            )

        except Exception:
            logger.exception(
                "An unexpected error occurred during tool call extraction."
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _update_thinking_state(self, text: str) -> None:
        """
        Update the thinking tag state based on text content.

        Args:
            text: Text to analyze for thinking tags
        """
        open_count = text.count("<think>")
        close_count = text.count("</think>")
        self.in_thinking_tag = open_count > close_count or (
            open_count == close_count and text.endswith("</think>")
        )

    def _is_potential_tag_start(self, text: str) -> bool:
        """
        Check if text might be the start of a tool call tag.

        Args:
            text: Text to check

        Returns:
            True if text could be the start of a tool call tag
        """
        for tag in [self.tool_call_start_token, self.tool_call_end_token]:
            if any(
                tag.startswith(text[-i:])
                for i in range(1, min(len(text) + 1, len(tag)))
            ):
                return True
        return False

    def _should_buffer_content(self, delta_text: str) -> bool:
        """
        Determine if content should be buffered for later processing.

        Args:
            delta_text: Delta text to check

        Returns:
            True if content should be buffered
        """
        if self.in_thinking_tag:
            return False
        return bool(
            self.pending_buffer
            or self.tool_call_start_token in delta_text
            or self.tool_call_end_token in delta_text
            or delta_text.startswith("<")
        )

    def _split_content_for_buffering(self, delta_text: str) -> tuple[str, str]:
        """
        Split delta text into safe content and potential tag content.

        Args:
            delta_text: Delta text to split

        Returns:
            Tuple of (safe_content, potential_tag_content)
        """
        if self.in_thinking_tag:
            return delta_text, ""

        for tag in [self.tool_call_start_token, self.tool_call_end_token]:
            for i in range(1, len(tag)):
                tag_prefix = tag[:i]
                pos = delta_text.rfind(tag_prefix)
                if pos != -1 and tag.startswith(delta_text[pos:]):
                    return delta_text[:pos], delta_text[pos:]
        return delta_text, ""

    def _process_buffer(self, new_content: str) -> str:
        """
        Process buffered content and return output content.

        Args:
            new_content: New content to add to buffer

        Returns:
            Processed output content
        """
        self.pending_buffer += new_content
        output_content = ""

        if self.in_thinking_tag:
            output_content = self.pending_buffer
            self.pending_buffer = ""
            return output_content

        while self.pending_buffer:
            start_pos = self.pending_buffer.find(self.tool_call_start_token)
            end_pos = self.pending_buffer.find(self.tool_call_end_token)

            if start_pos != -1 and (end_pos == -1 or start_pos < end_pos):
                tag_pos, tag_len = start_pos, len(self.tool_call_start_token)
            elif end_pos != -1:
                tag_pos, tag_len = end_pos, len(self.tool_call_end_token)
            else:
                if self._is_potential_tag_start(self.pending_buffer):
                    break
                output_content += self.pending_buffer
                self.pending_buffer = ""
                break

            output_content += self.pending_buffer[:tag_pos]
            self.pending_buffer = self.pending_buffer[tag_pos + tag_len :]

        return output_content

    def _reset_streaming_state(self) -> None:
        """Reset the streaming state to initial values."""
        self.streaming_state = {
            "current_tool_index": -1,
            "tool_ids": [],
            "sent_tools": [],
        }

    def _advance_to_next_tool(self) -> None:
        """Advance to the next tool in the streaming sequence."""
        self.streaming_state["current_tool_index"] = (
            int(self.streaming_state["current_tool_index"]) + 1
        )

    def _set_current_tool_index(self, index: int) -> None:
        """
        Set the current tool index.

        Args:
            index: Tool index to set
        """
        self.streaming_state["current_tool_index"] = index

    def _get_current_tool_index(self) -> int:
        """
        Get the current tool index.

        Returns:
            Current tool index
        """
        return int(self.streaming_state["current_tool_index"])

    def _get_next_unsent_tool_index(self, tool_count: int) -> int:
        """
        Get the index of the next unsent tool.

        Args:
            tool_count: Total number of tools

        Returns:
            Index of next unsent tool, or -1 if all tools sent
        """
        sent_tools = list(self.streaming_state["sent_tools"])
        for i in range(tool_count):
            if i < len(sent_tools):
                if not sent_tools[i]["sent_name"]:
                    return i
            else:
                return i
        return -1

    def _ensure_state_arrays(self, tool_count: int) -> None:
        """
        Ensure state arrays have sufficient capacity for tool_count tools.

        Args:
            tool_count: Number of tools to prepare for
        """
        sent_tools = list(self.streaming_state["sent_tools"])
        tool_ids = list(self.streaming_state["tool_ids"])

        while len(sent_tools) < tool_count:
            sent_tools.append(
                {
                    "sent_name": False,
                    "sent_arguments": "",
                    "id": make_tool_call_id(),
                }
            )

        while len(tool_ids) < tool_count:
            tool_ids.append(None)

        self.streaming_state["sent_tools"] = sent_tools
        self.streaming_state["tool_ids"] = tool_ids

    def _detect_tools_in_text(self, text: str) -> int:
        """
        Detect the number of tools in text by counting name patterns.

        Args:
            text: Text to analyze

        Returns:
            Number of tools detected
        """
        matches = self.tool_name_pattern.findall(text)
        return len(matches)

    def _find_tool_boundaries(self, text: str) -> list[tuple[int, int]]:
        """
        Find the boundaries of tool calls in text.

        Args:
            text: Text to analyze

        Returns:
            List of (start, end) positions for tool calls
        """
        boundaries = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                start = i
                depth = 0
                has_name = False
                has_arguments = False

                while i < len(text):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            segment = text[start:end]
                            if '"name"' in segment and '"arguments"' in segment:
                                boundaries.append((start, end))
                            break

                    if not has_name and '"name"' in text[start : i + 1]:
                        has_name = True
                    if not has_arguments and '"arguments"' in text[start : i + 1]:
                        has_arguments = True

                    i += 1

                if depth > 0 and has_name:
                    boundaries.append((start, i))
            else:
                i += 1
        return boundaries

    def _extract_tool_args(self, tool_content: str, args_match: re.Match[str]) -> str:
        """
        Extract tool arguments from tool content.

        Args:
            tool_content: Tool call content
            args_match: Regex match for arguments pattern

        Returns:
            Extracted arguments as string
        """
        args_start_pos = args_match.end()
        remaining_content = tool_content[args_start_pos:]

        if remaining_content.strip().startswith("{"):
            depth = 0
            for i, char in enumerate(remaining_content):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return remaining_content[: i + 1]
        else:
            args_end = remaining_content.find("}")
            if args_end > 0:
                return remaining_content[:args_end].strip()

        return remaining_content.rstrip("}").strip()

    def _get_current_tool_content(
        self, text: str, tool_index: int
    ) -> tuple[str | None, str | None]:
        """
        Get the content of a specific tool by index.

        Args:
            text: Text containing tool calls
            tool_index: Index of tool to extract

        Returns:
            Tuple of (tool_name, tool_arguments) or (None, None) if not found
        """
        boundaries = self._find_tool_boundaries(text)

        if tool_index >= len(boundaries):
            return None, None

        start, end = boundaries[tool_index]
        tool_content = text[start:end]

        name_match = self.tool_name_pattern.search(tool_content)
        name = name_match.group(1) if name_match else None

        args_match = self.tool_args_pattern.search(tool_content)
        if args_match:
            try:
                args_text = self._extract_tool_args(tool_content, args_match)
                return name, args_text
            except Exception:
                remaining_content = tool_content[args_match.end() :]
                args_text = remaining_content.rstrip("}").strip()
                return name, args_text

        return name, None

    def _handle_tool_name_streaming(
        self, tool_content: str, tool_count: int
    ) -> DeltaMessage | None:
        """
        Handle streaming of tool names.

        Args:
            tool_content: Content containing tool calls
            tool_count: Total number of tools

        Returns:
            DeltaMessage with tool name or None if no tool to stream
        """
        next_idx = self._get_next_unsent_tool_index(tool_count)

        if next_idx == -1:
            return None

        boundaries = self._find_tool_boundaries(tool_content)
        if next_idx >= len(boundaries):
            return None

        tool_name, _ = self._get_current_tool_content(tool_content, next_idx)
        if not tool_name:
            return None

        self._set_current_tool_index(next_idx)
        sent_tools = list(self.streaming_state["sent_tools"])
        tool_ids = list(self.streaming_state["tool_ids"])

        tool_id = sent_tools[next_idx]["id"]
        tool_ids[next_idx] = tool_id
        sent_tools[next_idx]["sent_name"] = True

        self.streaming_state["sent_tools"] = sent_tools
        self.streaming_state["tool_ids"] = tool_ids

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=next_idx,
                    type="function",
                    id=tool_id,
                    function=DeltaFunctionCall(name=tool_name).model_dump(
                        exclude_none=True
                    ),
                )
            ]
        )

    def _handle_tool_args_streaming(
        self, tool_content: str, tool_count: int
    ) -> DeltaMessage | None:
        """
        Handle streaming of tool arguments.

        Args:
            tool_content: Content containing tool calls
            tool_count: Total number of tools

        Returns:
            DeltaMessage with tool arguments or None if no arguments to stream
        """
        current_idx = self._get_current_tool_index()

        if current_idx < 0 or current_idx >= tool_count:
            return None

        tool_name, tool_args = self._get_current_tool_content(tool_content, current_idx)
        if not tool_name or tool_args is None:
            return None

        sent_tools = list(self.streaming_state["sent_tools"])

        if not sent_tools[current_idx]["sent_name"]:
            return None

        clean_args = self._clean_duplicate_braces(tool_args)
        sent_args = sent_tools[current_idx]["sent_arguments"]

        if clean_args != sent_args:
            if sent_args and clean_args.startswith(sent_args):
                args_delta = extract_intermediate_diff(clean_args, sent_args)
                if args_delta:
                    args_delta = self._clean_delta_braces(args_delta)
                    sent_tools[current_idx]["sent_arguments"] = clean_args
                    self.streaming_state["sent_tools"] = sent_tools

                    if clean_args.endswith("}"):
                        self._advance_to_next_tool()

                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=current_idx,
                                function=DeltaFunctionCall(
                                    arguments=args_delta
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
            elif not sent_args and clean_args:
                clean_args_delta = self._clean_delta_braces(clean_args)
                sent_tools[current_idx]["sent_arguments"] = clean_args
                self.streaming_state["sent_tools"] = sent_tools

                if clean_args.endswith("}"):
                    self._advance_to_next_tool()

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=current_idx,
                            function=DeltaFunctionCall(
                                arguments=clean_args_delta
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )

        return None

    def _is_end_tool_calls(self, current_text: str) -> bool:
        if self.tool_call_end_token not in current_text:
            return False

        end_token_positions = []
        search_start = 0
        while True:
            pos = current_text.find(self.tool_call_end_token, search_start)
            if pos == -1:
                break
            end_token_positions.append(pos)
            search_start = pos + 1

        think_regions = []
        for match in re.finditer(
            self.thinking_tag_pattern, current_text, flags=re.DOTALL
        ):
            think_regions.append((match.start(), match.end()))

        for pos in end_token_positions:
            in_think = any(
                pos >= t_start and pos < t_end for t_start, t_end in think_regions
            )
            if not in_think:
                return True

        return False

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
        self._update_thinking_state(current_text)

        if self.in_thinking_tag:
            return DeltaMessage(content=delta_text)

        if self._should_buffer_content(delta_text):
            buffered_output = self._process_buffer(delta_text)
            return DeltaMessage(content=buffered_output) if buffered_output else None

        if self._is_end_tool_calls(current_text):
            return DeltaMessage(content=delta_text)

        safe_content, potential_tag = self._split_content_for_buffering(delta_text)
        if potential_tag:
            self.pending_buffer += potential_tag
            return DeltaMessage(content=safe_content) if safe_content else None

        processed_current_text = self.preprocess_model_output(current_text)

        if self.tool_call_start_token not in processed_current_text:
            if (
                self.tool_call_end_token in delta_text
                and self.tool_call_start_token in current_text
            ):
                return None
            if delta_text.strip() == "" and self.tool_call_start_token in current_text:
                return None
            if (
                self._get_current_tool_index() != -1
                and self.tool_call_end_token in current_text
            ):
                self._reset_streaming_state()
            return DeltaMessage(content=delta_text)

        if (
            self.tool_call_start_token_id is not None
            and self.tool_call_start_token_id in delta_token_ids
            and len(delta_token_ids) == 1
        ):
            return None

        original_tool_start = self._find_tool_start_outside_thinking(current_text)
        if original_tool_start is None:
            return None

        content_before_tools = self._extract_content_before_tools(
            current_text, delta_text, original_tool_start
        )
        if content_before_tools:
            return DeltaMessage(content=content_before_tools)

        try:
            tool_content = self._extract_tool_content(current_text, original_tool_start)
            current_tools_count = self._detect_tools_in_text(tool_content)

            if current_tools_count == 0:
                return None

            if self._get_current_tool_index() == -1:
                self._reset_streaming_state()

            self._ensure_state_arrays(current_tools_count)

            return self._handle_tool_name_streaming(
                tool_content, current_tools_count
            ) or self._handle_tool_args_streaming(tool_content, current_tools_count)

        except Exception:
            logger.exception(
                "An unexpected error occurred ", "during streaming tool call handling."
            )
            return None

    def _find_tool_start_outside_thinking(self, current_text: str) -> int | None:
        """
        Find the start position of tool calls outside of thinking tags.

        Args:
            current_text: Current text to search

        Returns:
            Position of tool call start or None if not found
        """
        search_start = 0
        while True:
            pos = current_text.find(self.tool_call_start_token, search_start)
            if pos == -1:
                return None

            think_regions = [
                (m.start(), m.end())
                for m in re.finditer(
                    r"<think>(.*?)</think>", current_text, flags=re.DOTALL
                )
            ]
            in_think = any(
                pos >= t_start and pos < t_end for t_start, t_end in think_regions
            )

            if not in_think:
                return pos

            search_start = pos + 1

    def _extract_content_before_tools(
        self, current_text: str, delta_text: str, tool_start: int
    ) -> str | None:
        """
        Extract content that appears before tool calls.

        Args:
            current_text: Current text
            delta_text: Delta text
            tool_start: Start position of tools

        Returns:
            Content before tools or None
        """
        if tool_start > 0:
            delta_start_pos = len(current_text) - len(delta_text)
            if delta_start_pos < tool_start:
                content_part = delta_text
                if delta_start_pos + len(delta_text) > tool_start:
                    content_part = delta_text[: tool_start - delta_start_pos]
                return content_part if content_part else None
        return None

    def _extract_tool_content(self, current_text: str, tool_start: int) -> str:
        """
        Extract tool content from current text starting at tool_start.

        Args:
            current_text: Current text
            tool_start: Start position of tool calls

        Returns:
            Extracted tool content
        """
        tool_content_start = tool_start + len(self.tool_call_start_token)
        tool_content = current_text[tool_content_start:]

        end_pos = tool_content.find(self.tool_call_end_token)
        if end_pos != -1:
            tool_content = tool_content[:end_pos]

        return tool_content
