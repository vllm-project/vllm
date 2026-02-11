# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class Step3p5ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
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
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function(?:=|\s+)?(.*?)</function>", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Step3p5 RL Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

        # Get EOS token ID for EOS detection
        self.eos_token_id = getattr(self.model_tokenizer, "eos_token_id", None)

        logger.info(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def parser_should_check_for_unstreamed_tool_arg_tokens(self) -> bool:
        """
        Skip the remaining_call calculation in serving
        """
        return False

    def _reset_streaming_state(self):
        """Reset all streaming state for a new request."""
        self._processed_length: int = 0  # Position of last processed character
        self._tool_call_index: int = 0  # Number of tool calls processed so far
        self.streaming_request = None  # Current request being processed

    def _get_arguments_config(
        self, func_name: str, tools: list[ChatCompletionToolsParam] | None
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            if not hasattr(config, "type") or not (
                hasattr(config, "function") and hasattr(config.function, "name")
            ):
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
        logger.warning("Tool '%s' is not defined in the tools list.", func_name)
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the "
                    "string value.",
                    param_name,
                    func_name,
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                try:
                    float_value = float(param_value)
                    if float_value.is_integer():
                        return int(float_value)
                except (ValueError, TypeError):
                    pass
                try:
                    literal_value = ast.literal_eval(param_value)
                    if isinstance(literal_value, bool):
                        return int(literal_value)
                    if isinstance(literal_value, (int, float)):
                        return (
                            int(literal_value)
                            if float(literal_value).is_integer()
                            else literal_value
                        )
                except (ValueError, SyntaxError, TypeError):
                    pass
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not an integer "
                    "in tool '%s', returning raw string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except (ValueError, TypeError):
                try:
                    literal_value = ast.literal_eval(param_value)
                    if isinstance(literal_value, (int, float)):
                        return (
                            float(literal_value)
                            if float(literal_value) - int(float(literal_value)) != 0
                            else int(float(literal_value))
                        )
                except (ValueError, SyntaxError, TypeError):
                    pass
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not a float "
                    "in tool '%s', returning raw string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            normalized_value = param_value.strip().lower()
            if normalized_value in ["true", "false"]:
                return normalized_value == "true"
            if normalized_value in ["1", "0"]:
                return normalized_value == "1"
            try:
                literal_value = ast.literal_eval(param_value)
                if isinstance(literal_value, bool):
                    return literal_value
            except (ValueError, SyntaxError, TypeError):
                pass
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not a boolean "
                "in tool '%s', returning raw string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except (json.JSONDecodeError, TypeError, ValueError):
                    try:
                        literal_value = ast.literal_eval(param_value)
                        if isinstance(literal_value, (list, dict)):
                            return literal_value
                        if isinstance(literal_value, (tuple, set)):
                            return list(literal_value)
                    except (ValueError, SyntaxError, TypeError):
                        pass
                    logger.warning(
                        "Parsed value '%s' of parameter '%s' cannot be parsed "
                        "as JSON in tool '%s', returning raw string.",
                        param_value,
                        param_name,
                        func_name,
                    )
                    return param_value
            try:
                literal_value = ast.literal_eval(param_value)  # safer
                if isinstance(literal_value, (tuple, set)):
                    return list(literal_value)
                if (
                    isinstance(literal_value, (list, dict, str, int, float, bool))
                    or literal_value is None
                ):
                    return literal_value
            except (ValueError, SyntaxError, TypeError):
                pass
            logger.warning(
                "Parsed value '%s' of parameter '%s' cannot be converted via "
                "Python `ast.literal_eval()` in tool '%s', returning raw string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value

    def _parse_parameters_fallback(
        self,
        parameters: str,
        allowed_param_names: set[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Fallback parser for malformed parameter tags."""
        param_pairs: list[tuple[str, str]] = []
        pos = 0
        while True:
            start = parameters.find(self.parameter_prefix, pos)
            if start == -1:
                break
            name_start = start + len(self.parameter_prefix)
            name_end = parameters.find(">", name_start)
            if name_end == -1:
                newline_idx = parameters.find("\n", name_start)
                end_tag = parameters.find(self.parameter_end_token, name_start)
                next_param = parameters.find(self.parameter_prefix, name_start)
                candidates = [
                    idx for idx in [newline_idx, end_tag, next_param] if idx != -1
                ]
                if not candidates:
                    break
                name_end = min(candidates)
                value_start = name_end
            else:
                value_start = name_end + 1
            param_name = parameters[name_start:name_end].strip()
            next_param = parameters.find(self.parameter_prefix, value_start)
            end_tag = parameters.find(self.parameter_end_token, value_start)
            if end_tag == -1 or (next_param != -1 and next_param < end_tag):
                end = next_param if next_param != -1 else len(parameters)
                pos = end
            else:
                end = end_tag
                pos = end + len(self.parameter_end_token)
            param_value = parameters[value_start:end]
            if allowed_param_names is None or param_name in allowed_param_names:
                param_pairs.append((param_name, param_value))
        return param_pairs

    def _is_valid_json_arguments(self, arguments: str) -> bool:
        """Check if arguments can be loaded as JSON."""
        try:
            json.loads(arguments)
        except Exception:
            return False
        return True

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[ChatCompletionToolsParam] | None
    ) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.index(">")

        # check empty function name
        function_name = function_call_str[:end_index].strip()
        if function_name.startswith("="):
            function_name = function_name.lstrip("=").strip()
        if not function_name or function_name.strip("'\"") == "":
            logger.warning("Empty function name in tool call.")
            return None
        if function_name[0] in "\"'" and function_name[-1] == function_name[0]:
            function_name = function_name[1:-1].strip()
            if not function_name:
                logger.warning("Empty function name in tool call.")
                return None

        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        match_texts = self.tool_call_parameter_regex.findall(parameters)
        use_fallback = False
        if match_texts:
            for match_text in match_texts:
                if self.parameter_prefix in match_text or ">" not in match_text:
                    use_fallback = True
                    break
        else:
            use_fallback = self.parameter_prefix in parameters

        if use_fallback:
            allowed_param_names = (
                set(param_config.keys())
                if isinstance(param_config, dict) and param_config
                else None
            )
            param_pairs = self._parse_parameters_fallback(
                parameters, allowed_param_names
            )
        else:
            param_pairs = []
            for match_text in match_texts:
                idx = match_text.index(">")
                param_name = match_text[:idx]
                param_value = str(match_text[idx + 1 :])
                param_pairs.append((param_name, param_value))

        for param_name, param_value in param_pairs:
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )

        try:
            arguments = json.dumps(param_dict, ensure_ascii=False)
        except Exception as e:
            logger.warning("Error in converting parameter value: %s", e)
            return None
        return ToolCall(
            type="function",
            function=FunctionCall(name=function_name, arguments=arguments),
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
            function_matches = self.tool_call_function_regex.findall(tool_call_content)
            if len(function_matches) > 0:
                has_valid_function = True
            # Check if there's an unclosed function tag
            if (
                self.tool_call_prefix in tool_call_content
                and self.function_end_token not in tool_call_content
            ):
                return False

        if not has_valid_function:
            return False

        # Check 3: If parameter tags exist, they must be closed and correct
        for tool_call_content in tool_call_matches:
            # Count opening and closing parameter tags
            param_open_count = tool_call_content.count(self.parameter_prefix)
            param_close_count = tool_call_content.count(self.parameter_end_token)

            # If there are parameter tags, they must be balanced
            if param_open_count > 0:
                if param_open_count != param_close_count:
                    return False
                # Check if all parameter tags are properly closed using regex
                param_matches = self.tool_call_parameter_regex.findall(
                    tool_call_content
                )
                if len(param_matches) != param_open_count:
                    return False

        return True

    def _wrap_missing_tool_call_tags(self, model_output: str) -> str:
        """Wrap bare <function=...></function> blocks with <tool_call> tags."""
        if (
            self.tool_call_prefix not in model_output
            or self.function_end_token not in model_output
        ):
            return model_output

        def _wrap_bare_functions(text: str) -> str:
            pos = 0
            wrapped_parts: list[str] = []
            while True:
                func_idx = text.find(self.tool_call_prefix, pos)
                if func_idx == -1:
                    wrapped_parts.append(text[pos:])
                    break
                end_idx = text.find(self.function_end_token, func_idx)
                if end_idx == -1:
                    wrapped_parts.append(text[pos:])
                    break
                end_idx += len(self.function_end_token)
                wrapped_parts.append(text[pos:func_idx])
                wrapped_parts.append(self.tool_call_start_token)
                wrapped_parts.append(text[func_idx:end_idx])
                wrapped_parts.append(self.tool_call_end_token)

                ws_idx = end_idx
                while ws_idx < len(text) and text[ws_idx].isspace():
                    ws_idx += 1
                if text.startswith(self.tool_call_end_token, ws_idx):
                    if ws_idx > end_idx:
                        wrapped_parts.append(text[end_idx:ws_idx])
                    pos = ws_idx + len(self.tool_call_end_token)
                else:
                    pos = end_idx
            return "".join(wrapped_parts)

        tool_call_ranges = [
            match.span()
            for match in self.tool_call_complete_regex.finditer(model_output)
        ]
        if not tool_call_ranges:
            return _wrap_bare_functions(model_output)

        wrapped_parts: list[str] = []
        pos = 0
        for start, end in tool_call_ranges:
            if start < pos:
                continue
            wrapped_parts.append(_wrap_bare_functions(model_output[pos:start]))
            wrapped_parts.append(model_output[start:end])
            pos = end
        wrapped_parts.append(_wrap_bare_functions(model_output[pos:]))
        return "".join(wrapped_parts)

    def _normalize_prev_arguments(self, args_value: Any) -> Any:
        if isinstance(args_value, str):
            try:
                return json.loads(args_value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return args_value
        return args_value

    def _update_prev_tool_call_state(self, tool_calls: list[ToolCall]) -> None:
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        for tool_call in tool_calls:
            if not tool_call or not tool_call.function:
                continue
            args_value = tool_call.function.arguments
            if isinstance(args_value, str):
                args_json = args_value
            elif args_value is None:
                args_json = ""
            else:
                try:
                    args_json = json.dumps(args_value, ensure_ascii=False)
                except (TypeError, ValueError):
                    args_json = str(args_value)

            prev_args = self._normalize_prev_arguments(args_json)
            self.prev_tool_call_arr.append(
                {
                    "name": tool_call.function.name,
                    "arguments": prev_args,
                }
            )
            try:
                expected_args_json = json.dumps(prev_args, ensure_ascii=False)
            except (TypeError, ValueError):
                expected_args_json = args_json

            # Serving may subtract the latest delta length from
            # streamed_args_for_tool to detect unstreamed suffixes. Since this
            # parser emits full arguments at once, store expected+actual so
            # the subtraction yields expected_args_json and no resend occurs.
            self.streamed_args_for_tool.append(expected_args_json + args_json)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        try:
            origin_model_output = model_output
            try:
                # Fallback: handle outputs without <tool_call> wrapper.
                origin_model_output = self._wrap_missing_tool_call_tags(
                    origin_model_output
                )
                model_output = origin_model_output
            except Exception:
                pass

            # Use streaming-like approach: process position by position
            valid_tool_calls = []
            content_parts = []
            processed_length = 0

            while processed_length < len(model_output):
                # Find next tool call start
                tool_start_idx = self._find_tool_call_start(
                    model_output, processed_length
                )

                # Case 1: No more tool calls - add remaining as content
                if tool_start_idx == -1:
                    remaining = model_output[processed_length:]
                    if remaining:
                        content_parts.append(remaining)
                    break

                # Case 2: Content before tool call
                if tool_start_idx > processed_length:
                    content_before = model_output[processed_length:tool_start_idx]
                    # Skip whitespace-only content between tool calls
                    # Check if we just ended a tool call and this is pure whitespace
                    if processed_length > 0:
                        text_before = model_output[:processed_length]
                        if (
                            text_before.rstrip().endswith(self.tool_call_end_token)
                            and content_before.strip() == ""
                        ):
                            # Skip whitespace between tool calls
                            pass
                        else:
                            content_parts.append(content_before)
                    else:
                        content_parts.append(content_before)

                # Case 3: Try to find complete tool call
                tool_end_idx = self._find_first_complete_tool_call_end(
                    model_output, tool_start_idx
                )

                # If tool call is incomplete - add remaining as content and stop
                if tool_end_idx == -1:
                    remaining = model_output[tool_start_idx:]
                    if remaining:
                        content_parts.append(remaining)
                    break

                # Extract and try to parse the complete tool call
                tool_call_text = model_output[tool_start_idx:tool_end_idx]
                parsed_result = self.extract_tool_calls_basic(tool_call_text, request)

                # If parsing succeeded, record the tool call(s)
                if parsed_result.tools_called and parsed_result.tool_calls:
                    valid_tool_calls.extend(parsed_result.tool_calls)
                    processed_length = tool_end_idx
                else:
                    # Parsing failed - treat this tool call as content
                    content_parts.append(tool_call_text)
                    processed_length = tool_end_idx

            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self._update_prev_tool_call_state(valid_tool_calls)

            # Combine content parts
            content = "".join(content_parts) if content_parts else None

            return ExtractedToolCallInformation(
                tools_called=(len(valid_tool_calls) > 0),
                tool_calls=valid_tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.warning("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_basic(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        model_output = self._wrap_missing_tool_call_tags(model_output)
        # Quick check to avoid unnecessary processing
        if not self._check_format(model_output):
            tool_call_matches = self.tool_call_complete_regex.findall(model_output)
            if len(tool_call_matches) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls: list[ToolCall] = []
            for function_call_str in function_calls:
                tool_call = self._parse_xml_function_call(
                    function_call_str, request.tools
                )
                if tool_call:
                    tool_calls.append(tool_call)
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )
            for tool_call in tool_calls:
                if (
                    not tool_call.function
                    or tool_call.function.arguments is None
                    or not self._is_valid_json_arguments(tool_call.function.arguments)
                ):
                    logger.warning(
                        "Invalid JSON arguments in tool call, falling back to content."
                    )
                    return ExtractedToolCallInformation(
                        tools_called=False, tool_calls=[], content=model_output
                    )

            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self._update_prev_tool_call_state(tool_calls)

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
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _find_first_complete_tool_call_end(self, text: str, start_pos: int = 0) -> int:
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
        end_idx = text.find(
            self.tool_call_end_token, start_idx + len(self.tool_call_start_token)
        )
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
            content = text[end_pos:next_start] if next_start != -1 else text[end_pos:]

            # Store content (empty string if whitespace-only)
            content_segments.append(content if content.strip() else "")

            if next_start == -1:
                break
            pos = next_start

        return content_segments

    def _convert_tool_calls_to_deltas(
        self, tool_calls: list[ToolCall], starting_index: int = 0
    ) -> list[DeltaToolCall]:
        """Convert complete ToolCall list to DeltaToolCall list.

        Returns complete tool calls without splitting into fragments.

        Args:
            tool_calls: List of tool calls to convert
            starting_index: Starting index for tool calls (default 0)

        Returns:
            List of DeltaToolCall with complete arguments
        """
        delta_tool_calls = []
        for i, tool_call in enumerate[ToolCall](tool_calls):
            index = starting_index + i
            tool_id = self._generate_tool_call_id()

            # Create complete DeltaToolCall with full arguments
            delta_tool_calls.append(
                DeltaToolCall(
                    index=index,
                    id=tool_id,
                    function=DeltaFunctionCall(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                    type="function",
                )
            )

        return delta_tool_calls

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
        has_eos = (
            self.eos_token_id is not None
            and delta_token_ids
            and self.eos_token_id in delta_token_ids
        )

        # If no delta text, check if we need to return empty delta for finish_reason
        if not delta_text and not has_eos:
            # Check if this is an EOS token after all tool calls are complete
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token
                    ) - current_text.count(self.tool_call_end_token)
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
                    self.tool_call_start_token
                ) - current_text.count(self.tool_call_end_token)
                if open_calls == 0:
                    accumulated_deltas.append(DeltaMessage(content=""))

        # Return results
        return self._format_delta_result(accumulated_deltas)

    def _has_unprocessed_content(self, current_text: str) -> bool:
        """Check if there's unprocessed content in the buffer."""
        return self._processed_length < len(current_text)

    def _process_next_chunk(
        self, current_text: str
    ) -> DeltaMessage | list[DeltaMessage] | None:
        """Process next chunk: either regular content or a complete tool call.

        Args:
            current_text: Current accumulated text

        Returns:
            - DeltaMessage or list of DeltaMessage if processed successfully
            - None if cannot proceed (need more tokens)
        """
        # Find next tool call start
        tool_start_idx = self._find_tool_call_start(
            current_text, self._processed_length
        )

        # Case 1: No tool call found - return remaining content
        if tool_start_idx == -1:
            return self._process_content(
                current_text, self._processed_length, len(current_text)
            )

        # Case 2: Content before tool call
        if tool_start_idx > self._processed_length:
            return self._process_content(
                current_text, self._processed_length, tool_start_idx
            )

        # Case 3: Tool call at current position
        # Find end of the first complete tool call
        tool_end_idx = self._find_first_complete_tool_call_end(
            current_text, tool_start_idx
        )

        if tool_end_idx == -1:
            # Tool call incomplete, wait for more tokens
            return None

        # Process complete tool call
        return self._process_complete_tool_calls(
            current_text, tool_start_idx, tool_end_idx
        )

    def _process_content(
        self, current_text: str, start_pos: int, end_pos: int
    ) -> DeltaMessage | None:
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
        if start_pos > 0:
            # Check if text before start_pos ends with </tool_call>
            text_before = current_text[:start_pos]
            if (
                text_before.rstrip().endswith(self.tool_call_end_token)
                and content.strip() == ""
            ):
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

    def _flush_remaining_content(self, current_text: str) -> DeltaMessage | None:
        """Flush any remaining unprocessed content as regular content.

        Args:
            current_text: Current accumulated text

        Used when EOS token is encountered to handle incomplete tool calls.
        """
        if not self._has_unprocessed_content(current_text):
            return None

        remaining = current_text[self._processed_length :]
        if remaining:
            self._processed_length = len(current_text)
            return DeltaMessage(content=remaining)

        self._processed_length = len(current_text)
        return None

    def _format_delta_result(self, deltas: list[DeltaMessage]) -> DeltaMessage | None:
        """Format delta result for return.

        Merges all deltas into a single DeltaMessage.

        Args:
            deltas: List of delta messages

        Returns:
            - None if empty
            - Single merged DeltaMessage with all content and tool_calls
        """
        if not deltas:
            return None

        if len(deltas) == 1:
            return deltas[0]

        # Merge multiple deltas into one
        merged_content_parts = []
        merged_tool_calls = []

        for delta in deltas:
            if delta.content:
                merged_content_parts.append(delta.content)
            if delta.tool_calls:
                merged_tool_calls.extend(delta.tool_calls)

        # Create merged DeltaMessage
        merged_content = "".join(merged_content_parts) if merged_content_parts else None

        # Build kwargs - only include tool_calls if non-empty
        kwargs: dict[str, Any] = {"content": merged_content}
        if merged_tool_calls:
            kwargs["tool_calls"] = merged_tool_calls

        return DeltaMessage(**kwargs)

    def _process_complete_tool_calls(
        self, current_text: str, start_pos: int, end_pos: int
    ) -> list[DeltaMessage] | None:
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
            result = self.extract_tool_calls_basic(
                text_to_parse, self.streaming_request
            )

            # Case 1: Successfully parsed tool calls
            if result.tools_called and result.tool_calls:
                # Note: Due to _find_first_complete_tool_call_end, we typically
                # process only one tool call at a time
                # but we can also process multiple tool calls below
                deltas = self._build_tool_call_deltas(result.tool_calls, text_to_parse)
                self._update_state_after_tool_calls(result.tool_calls, end_pos)
                return deltas if deltas else None

            # Case 2: Parsing failed - treat as regular content
            self._processed_length = end_pos
            return [DeltaMessage(content=text_to_parse)]

        except Exception as e:
            # Exception during parsing - treat as content
            logger.debug("Failed to parse tool calls: %s, treating as content", e)
            self._processed_length = end_pos
            failed_text = current_text[start_pos:end_pos]
            return [DeltaMessage(content=failed_text)] if failed_text else None

    def _build_tool_call_deltas(
        self, tool_calls: list[ToolCall], parsed_text: str
    ) -> list[DeltaMessage]:
        """Build delta messages from parsed tool calls with interleaved content.

        Args:
            tool_calls: List of parsed tool calls
            parsed_text: Original text that was parsed

        Returns:
            List of DeltaMessage with tool calls and content interleaved
        """
        # Extract content segments between tool calls
        content_segments = self._extract_content_between_tool_calls_list(parsed_text)

        # Convert all tool calls to DeltaToolCall list
        delta_tool_calls = self._convert_tool_calls_to_deltas(
            tool_calls, self._tool_call_index
        )

        # Merge all content segments into a single string
        merged_content = "".join(content_segments)

        # Return a single DeltaMessage with all tool calls and content
        # Build kwargs - only include non-empty fields
        kwargs: dict[str, Any] = {}
        if merged_content:
            kwargs["content"] = merged_content
        if delta_tool_calls:
            kwargs["tool_calls"] = delta_tool_calls

        # Only return DeltaMessage if we have content or tool_calls
        if kwargs:
            return [DeltaMessage(**kwargs)]
        else:
            return []

    def _update_state_after_tool_calls(
        self, tool_calls: list[ToolCall], end_pos: int
    ) -> None:
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
        self._update_prev_tool_call_state(tool_calls)
