# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
import uuid
from collections.abc import Sequence
from typing import Any, Optional, Union

import regex as re

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionToolsParam,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


class Qwen3CoderToolParserRL(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: Optional[str] = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>", re.DOTALL)
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (self.tool_call_start_token_id is None
                or self.tool_call_end_token_id is None):
            raise RuntimeError(
                "Qwen3 XML Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")

        # Get EOS token ID for EOS detection
        self.eos_token_id = getattr(self.model_tokenizer, 'eos_token_id', None)

        logger.info("vLLM Successfully import tool parser %s !",
                    self.__class__.__name__)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state for a new request."""
        self._processed_length: int = 0  # Position of last processed character
        self._tool_call_index: int = 0  # Number of tool calls processed so far
        self.streaming_request = None  # Current request being processed

    def _get_arguments_config(
            self, func_name: str,
            tools: Optional[list[ChatCompletionToolsParam]]) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            if not hasattr(config, "type") or not (hasattr(
                    config, "function") and hasattr(config.function, "name")):
                continue
            if config.type == "function" and config.function.name == func_name:
                if not hasattr(config.function, "parameters"):
                    return {}
                params = config.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.warning("Tool '%s' is not defined in the tools list.",
                       func_name)
        return {}

    def _convert_param_value(self, param_value: str, param_name: str,
                             param_config: dict, func_name: str) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the "
                    "string value.", param_name, func_name)
            return param_value

        if isinstance(param_config[param_name],
                      dict) and "type" in param_config[param_name]:
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif param_type.startswith("int") or param_type.startswith(
                "uint") or param_type.startswith(
                    "long") or param_type.startswith(
                        "short") or param_type.startswith("unsigned"):
            try:
                return int(param_value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Parsed value '{param_value}' of parameter '{param_name}' "
                    f"is not an integer in tool '{func_name}'.") from e
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return float_param_value if float_param_value - int(
                    float_param_value) != 0 else int(float_param_value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Parsed value '{param_value}' of parameter '{param_name}' "
                    f"is not a float in tool '{func_name}'.") from e
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                raise ValueError(
                    f"Parsed value '{param_value}' of parameter '{param_name}' "
                    f"is not a boolean (`true` or `false`) in tool '{func_name}'."
                )
            return param_value == "true"
        else:
            if param_type in ["object", "array", "arr"
                              ] or param_type.startswith(
                                  "dict") or param_type.startswith("list"):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    raise ValueError(
                        f"Parsed value '{param_value}' of parameter '{param_name}' "
                        f"cannot be parsed with json.loads in tool '{func_name}'."
                    ) from e
            try:
                param_value = ast.literal_eval(param_value)  # safer
            except (ValueError, SyntaxError, TypeError) as e:
                raise ValueError(
                    f"Parsed value '{param_value}' of parameter '{param_name}' "
                    f"cannot be converted via Python `ast.literal_eval()` in tool "
                    f"'{func_name}'.") from e
            return param_value

    def _parse_xml_function_call(
            self, function_call_str: str,
            tools: Optional[list[ChatCompletionToolsParam]]
    ) -> Optional[ToolCall]:

        # Extract function name
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1:]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1:])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            try:
                param_dict[param_name] = self._convert_param_value(
                    param_value, param_name, param_config, function_name)
            except Exception:
                return None
        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name,
                                  arguments=json.dumps(param_dict,
                                                       ensure_ascii=False)),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls
        raw_tool_calls = self.tool_call_complete_regex.findall(model_output)

        # if no closed tool_call tags found, return empty list
        if len(raw_tool_calls) == 0:
            return []

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            function_matches = self.tool_call_function_regex.findall(tool_call)
            raw_function_calls.extend(function_matches)

        return raw_function_calls

    def _check_format(self, model_output: str) -> bool:
        """Check if model output contains properly formatted tool call.
        
        Requirements:
        1. Must have closed tool_call tags (<tool_call>...</tool_call>)
        2. Must have closed function tags (<function=...</function>)
        3. If parameter tags exist, they must be closed and correct
        
        Returns True if the format is valid, False otherwise.
        """
        # Check 1: Must have closed tool_call tags
        tool_call_matches = self.tool_call_complete_regex.findall(model_output)
        if len(tool_call_matches) == 0:
            return False

        # Check 2: Must have closed function tags within tool_call
        has_valid_function = False
        for tool_call_content in tool_call_matches:
            function_matches = self.tool_call_function_regex.findall(
                tool_call_content)
            if len(function_matches) > 0:
                has_valid_function = True
            # Check if there's an unclosed function tag
            if self.tool_call_prefix in tool_call_content and self.function_end_token not in tool_call_content:
                return False

        if not has_valid_function:
            return False

        # Check 3: If parameter tags exist, they must be closed and correct
        for tool_call_content in tool_call_matches:
            # Count opening and closing parameter tags
            param_open_count = tool_call_content.count(self.parameter_prefix)
            param_close_count = tool_call_content.count(
                self.parameter_end_token)

            # If there are parameter tags, they must be balanced
            if param_open_count > 0:
                if param_open_count != param_close_count:
                    return False
                # Check if all parameter tags are properly closed using regex
                param_matches = self.tool_call_parameter_regex.findall(
                    tool_call_content)
                if len(param_matches) != param_open_count:
                    return False

        return True

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Quick check to avoid unnecessary processing
        if not self._check_format(model_output):
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

            tool_calls = [
                self._parse_xml_function_call(function_call_str, request.tools)
                for function_call_str in function_calls
            ]

            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append({
                        "name":
                        tool_call.function.name,
                        "arguments":
                        tool_call.function.arguments,
                    })

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            content = model_output[:content_index]  # .rstrip()

            return ExtractedToolCallInformation(
                tools_called=(len(tool_calls) > 0),
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.warning("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def _find_first_complete_tool_call_end(self,
                                           text: str,
                                           start_pos: int = 0) -> int:
        """Find the end position of the first complete tool call.
        
        Args:
            text: Text to search in
            start_pos: Position to start searching from
            
        Returns:
            Position after the first </tool_call> tag, or -1 if incomplete
            
        Example:
            "<tool_call>...</tool_call>..." returns position after </tool_call>
        """
        # Find tool call start
        start_idx = text.find(self.tool_call_start_token, start_pos)
        if start_idx == -1:
            return -1

        # Find matching end token
        end_idx = text.find(self.tool_call_end_token,
                            start_idx + len(self.tool_call_start_token))
        if end_idx == -1:
            return -1  # Incomplete tool call

        # Return position after end token
        return end_idx + len(self.tool_call_end_token)

    def _find_tool_call_start(self, text: str, start_pos: int = 0) -> int:
        """Find the start position of next tool call.
        
        Args:
            text: Text to search in
            start_pos: Position to start searching from
            
        Returns:
            Position of <tool_call> token, or -1 if not found
        """
        return text.find(self.tool_call_start_token, start_pos)

    def _extract_content_between_tool_calls_list(self, text: str) -> list[str]:
        """Extract content segments after each tool call.
        
        For n tool calls, returns n segments where segment[i] is the content
        after tool_call[i] (before tool_call[i+1] or at the end).
        
        Empty or whitespace-only segments are represented as empty string "".
        
        Args:
            text: Text containing tool calls
            
        Returns:
            List of content segments (one per tool call)
        """
        content_segments = []
        pos = 0

        while True:
            # Find end of current tool call
            end_pos = text.find(self.tool_call_end_token, pos)
            if end_pos == -1:
                break

            # Move past the end token
            end_pos += len(self.tool_call_end_token)

            # Find start of next tool call
            next_start = self._find_tool_call_start(text, end_pos)

            # Extract content between current end and next start (or text end)
            content = text[end_pos:next_start] if next_start != -1 else text[
                end_pos:]

            # Store content (empty string if whitespace-only)
            content_segments.append(content if content.strip() else "")

            if next_start == -1:
                break
            pos = next_start

        return content_segments

    def _convert_tool_calls_to_deltas(
            self,
            tool_calls: list[ToolCall],
            starting_index: int = 0) -> list[DeltaMessage]:
        """Convert complete ToolCall list to delta message sequence.
        
        Format: header (function name) -> { -> param1 -> param2 -> ... -> }
        
        Args:
            tool_calls: List of tool calls to convert
            starting_index: Starting index for tool calls (default 0)
        """
        deltas = []
        for i, tool_call in enumerate(tool_calls):
            index = starting_index + i
            tool_id = self._generate_tool_call_id()

            # Header delta: function name
            deltas.append(
                DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=index,
                        id=tool_id,
                        function=DeltaFunctionCall(
                            name=tool_call.function.name, arguments=""),
                        type="function",
                    )
                ]))

            # Opening brace
            deltas.append(
                DeltaMessage(tool_calls=[
                    DeltaToolCall(index=index,
                                  function=DeltaFunctionCall(arguments="{"))
                ]))

            # Parse arguments JSON to extract parameters
            try:
                args_dict = json.loads(tool_call.function.arguments)
                param_names = list(args_dict.keys())

                for param_idx, param_name in enumerate(param_names):
                    param_value = args_dict[param_name]
                    serialized_value = json.dumps(param_value,
                                                  ensure_ascii=False)

                    if param_idx == 0:
                        json_fragment = f'"{param_name}": {serialized_value}'
                    else:
                        json_fragment = f', "{param_name}": {serialized_value}'

                    deltas.append(
                        DeltaMessage(tool_calls=[
                            DeltaToolCall(index=index,
                                          function=DeltaFunctionCall(
                                              arguments=json_fragment))
                        ]))
            except (json.JSONDecodeError, KeyError):
                # If parsing fails, just send the arguments as-is
                if tool_call.function.arguments:
                    deltas.append(
                        DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=index,
                                function=DeltaFunctionCall(
                                    arguments=tool_call.function.arguments))
                        ]))

            # Closing brace
            deltas.append(
                DeltaMessage(tool_calls=[
                    DeltaToolCall(index=index,
                                  function=DeltaFunctionCall(arguments="}"))
                ]))

        return deltas

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, list[DeltaMessage], None]:
        """Extract tool calls from streaming text using complete parsing.
        
        Strategy:
        1. Accumulate text in buffer and track processed position
        2. In each iteration, try to extract content or complete tool calls
        3. Parse complete tool calls using non-streaming method
        4. Convert parsed results to delta sequence
        5. Handle EOS token to flush incomplete tool calls as content
        """
        # Initialize state for new request
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # Check for EOS token
        has_eos = (self.eos_token_id is not None and delta_token_ids
                   and self.eos_token_id in delta_token_ids)

        # NOTE: The above simple check may incorrectly detect EOS when model output
        # contains <|im_end|> tokens (e.g., in multi-turn conversations).
        # If needed, use the more sophisticated check below:
        #
        # has_eos = False
        # if self.eos_token_id is not None and delta_token_ids and self.eos_token_id in delta_token_ids:
        #     if not delta_text:
        #         # Mode 1: Empty delta with EOS - definitely stream terminator
        #         has_eos = True
        #     elif delta_text:
        #         # Mode 2: Check if EOS is extra (not part of delta_text encoding)
        #         # Encode delta_text to see how many tokens it should produce
        #         encoded_delta = self.model_tokenizer.encode(delta_text, add_special_tokens=False)
        #         # If delta_token_ids has MORE tokens than encoded_delta,
        #         # the extra token is the EOS terminator
        #         if len(delta_token_ids) > len(encoded_delta):
        #             has_eos = True

        # If no delta text, check if we need to return empty delta for finish_reason
        if not delta_text and not has_eos:
            # Check if this is an EOS token after all tool calls are complete
            # Similar to qwen3coder_tool_parser.py logic
            if (delta_token_ids
                    and self.tool_call_end_token_id not in delta_token_ids):
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text))

                # If we have completed tool calls and populated prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token) - current_text.count(
                            self.tool_call_end_token)
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
            return None

        # Process all available content
        accumulated_deltas: list[DeltaMessage] = []

        while self._has_unprocessed_content(current_text):
            # Try to process next chunk (content or tool call)
            delta = self._process_next_chunk(current_text)

            if delta is None:
                # Cannot proceed further, need more tokens
                break

            # Accumulate deltas
            if isinstance(delta, list):
                accumulated_deltas.extend(delta)
            else:
                accumulated_deltas.append(delta)

        # Handle EOS: flush any remaining incomplete tool calls as content
        if has_eos:
            remaining_delta = self._flush_remaining_content(current_text)
            if remaining_delta:
                accumulated_deltas.append(remaining_delta)
            # If no remaining content but we have tool calls, return empty delta
            elif len(self.prev_tool_call_arr) > 0:
                # Check if all tool calls are closed
                open_calls = current_text.count(
                    self.tool_call_start_token) - current_text.count(
                        self.tool_call_end_token)
                if open_calls == 0:
                    accumulated_deltas.append(DeltaMessage(content=""))

        # Return results
        return self._format_delta_result(accumulated_deltas)

    def _has_unprocessed_content(self, current_text: str) -> bool:
        """Check if there's unprocessed content in the buffer."""
        return self._processed_length < len(current_text)

    def _process_next_chunk(
            self, current_text: str
    ) -> Union[DeltaMessage, list[DeltaMessage], None]:
        """Process next chunk: either regular content or a complete tool call.
        
        Args:
            current_text: Current accumulated text
            
        Returns:
            - DeltaMessage or list of DeltaMessage if processed successfully
            - None if cannot proceed (need more tokens)
        """
        # Find next tool call start
        tool_start_idx = self._find_tool_call_start(current_text,
                                                    self._processed_length)

        # Case 1: No tool call found - return remaining content
        if tool_start_idx == -1:
            return self._process_content(current_text, self._processed_length,
                                         len(current_text))

        # Case 2: Content before tool call
        if tool_start_idx > self._processed_length:
            return self._process_content(current_text, self._processed_length,
                                         tool_start_idx)

        # Case 3: Tool call at current position
        # Find end of the first complete tool call
        tool_end_idx = self._find_first_complete_tool_call_end(
            current_text, tool_start_idx)

        if tool_end_idx == -1:
            # Tool call incomplete, wait for more tokens
            return None

        # Process complete tool call
        return self._process_complete_tool_calls(current_text, tool_start_idx,
                                                 tool_end_idx)

    def _process_content(self, current_text: str, start_pos: int,
                         end_pos: int) -> Union[DeltaMessage, None]:
        """Process regular content (non-tool-call text).
        
        Args:
            current_text: Current accumulated text
            start_pos: Start position in buffer
            end_pos: End position in buffer
            
        Returns:
            DeltaMessage with content if non-empty
        """
        if start_pos >= end_pos:
            return None

        content = current_text[start_pos:end_pos]

        # Check if we're between tool calls - skip whitespace
        # Similar to qwen3coder_tool_parser.py logic
        if start_pos > 0:
            # Check if text before start_pos (after stripping trailing whitespace) ends with </tool_call>
            text_before = current_text[:start_pos]
            if (text_before.rstrip().endswith(self.tool_call_end_token)
                    and content.strip() == ""):
                # We just ended a tool call, skip whitespace between tool calls
                self._processed_length = end_pos
                return None

        # Return content if non-empty
        if content:
            self._processed_length = end_pos
            return DeltaMessage(content=content)

        # Mark as processed even if empty
        self._processed_length = end_pos
        return None

    def _flush_remaining_content(
            self, current_text: str) -> Union[DeltaMessage, None]:
        """Flush any remaining unprocessed content as regular content.
        
        Args:
            current_text: Current accumulated text
            
        Used when EOS token is encountered to handle incomplete tool calls.
        """
        if not self._has_unprocessed_content(current_text):
            return None

        remaining = current_text[self._processed_length:]
        if remaining:
            self._processed_length = len(current_text)
            return DeltaMessage(content=remaining)

        self._processed_length = len(current_text)
        return None

    def _format_delta_result(
        self, deltas: list[DeltaMessage]
    ) -> Union[DeltaMessage, list[DeltaMessage], None]:
        """Format delta result for return.
        
        Args:
            deltas: List of delta messages
            
        Returns:
            - None if empty
            - Single DeltaMessage if only one
            - List of DeltaMessage if multiple
        """
        if not deltas:
            return None
        elif len(deltas) == 1:
            return deltas[0]
        else:
            return deltas

    def _process_complete_tool_calls(
            self, current_text: str, start_pos: int,
            end_pos: int) -> Union[list[DeltaMessage], None]:
        """Process complete tool calls and convert to delta sequence.
        
        Args:
            current_text: Current accumulated text
            start_pos: Start position (should be at <tool_call>)
            end_pos: End position (after </tool_call>)
            
        Returns:
            List of DeltaMessage if successful, None otherwise
        """
        try:
            # Extract text segment containing complete tool call(s)
            text_to_parse = current_text[start_pos:end_pos]

            # Parse using non-streaming method
            result = self.extract_tool_calls(text_to_parse,
                                             self.streaming_request)

            # Case 1: Successfully parsed tool calls
            if result.tools_called and result.tool_calls:
                # Note: Due to _find_first_complete_tool_call_end, we typically
                # process only one tool call at a time
                # but we can also process multiple tool calls below
                deltas = self._build_tool_call_deltas(result.tool_calls,
                                                      text_to_parse)
                self._update_state_after_tool_calls(result.tool_calls, end_pos)
                return deltas if deltas else None

            # Case 2: Parsing failed - treat as regular content
            self._processed_length = end_pos
            return [DeltaMessage(content=text_to_parse)]

        except Exception as e:
            # Exception during parsing - treat as content
            logger.debug(
                f"Failed to parse tool calls: {e}, treating as content")
            self._processed_length = end_pos
            failed_text = current_text[start_pos:end_pos]
            return [DeltaMessage(content=failed_text)] if failed_text else None

    def _build_tool_call_deltas(self, tool_calls: list[ToolCall],
                                parsed_text: str) -> list[DeltaMessage]:
        """Build delta messages from parsed tool calls with interleaved content.
        
        Args:
            tool_calls: List of parsed tool calls
            parsed_text: Original text that was parsed
            
        Returns:
            List of DeltaMessage with tool calls and content interleaved
        """
        deltas = []

        # Extract content segments between tool calls
        content_segments = self._extract_content_between_tool_calls_list(
            parsed_text)

        # Build deltas: tool_call[i] -> content[i] (if exists)
        for i, tool_call in enumerate(tool_calls):
            # Convert tool call to delta sequence
            tool_deltas = self._convert_tool_calls_to_deltas(
                [tool_call], self._tool_call_index + i)
            deltas.extend(tool_deltas)

            # Add content after this tool call if exists
            if i < len(content_segments) and content_segments[i]:
                deltas.append(DeltaMessage(content=content_segments[i]))

        return deltas

    def _update_state_after_tool_calls(self, tool_calls: list[ToolCall],
                                       end_pos: int) -> None:
        """Update internal state after processing tool calls.
        
        Args:
            tool_calls: List of processed tool calls
            end_pos: End position in buffer
        """
        # Update processed position
        self._processed_length = end_pos

        # Update tool call index
        self._tool_call_index += len(tool_calls)

        # Update prev_tool_call_arr for finish_reason
        self.prev_tool_call_arr.clear()
        for tool_call in tool_calls:
            if tool_call:
                self.prev_tool_call_arr.append({
                    "name":
                    tool_call.function.name,
                    "arguments":
                    tool_call.function.arguments,
                })
