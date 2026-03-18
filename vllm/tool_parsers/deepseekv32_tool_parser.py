# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
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


class DeepSeekV32ToolParser(ToolParser):
    """
    example tool call content:
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []

        # Sentinel tokens
        self.dsml_token: str = "｜DSML｜"
        self.dsml_start_check: str = "<" + self.dsml_token
        self.tool_call_start_token: str = "<｜DSML｜function_calls>"
        self.tool_call_end_token: str = "</｜DSML｜function_calls>"
        self.invoke_start_prefix: str = "<｜DSML｜invoke name="
        self.invoke_end_token: str = "</｜DSML｜invoke>"
        self.parameter_prefix: str = "<｜DSML｜parameter name="
        self.parameter_end_token: str = "</｜DSML｜parameter>"

        # Streaming state
        self.is_tool_call_started: bool = False
        self.current_tool_index: int = 0

        # Regex patterns for complete parsing
        self.tool_call_complete_regex = re.compile(
            r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>", re.DOTALL
        )
        self.invoke_complete_regex = re.compile(
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>', re.DOTALL
        )
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(?:true|false)"\s*>(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def adjust_request(self, request):
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure tool call tokens
            # (<｜DSML｜function_calls>, </｜DSML｜function_calls>)
            # are not skippedduring decoding.
            # Even though they are not marked as special tokens,
            # setting skip_special_tokens=False ensures proper handling in
            # transformers 5.x where decoding behavior may have changed.
            request.skip_special_tokens = False
        return request

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    def _parse_invoke_params(self, invoke_str: str) -> dict | None:
        param_dict = dict()
        for param_name, param_val in self.parameter_complete_regex.findall(invoke_str):
            param_dict[param_name] = param_val
        return param_dict

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        # Quick check
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = []

            # Find all complete tool_call blocks
            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                # Find all invokes within this tool_call
                for invoke_name, invoke_content in self.invoke_complete_regex.findall(
                    tool_call_match
                ):
                    param_dict = self._parse_invoke_params(invoke_content)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=invoke_name,
                                arguments=json.dumps(param_dict, ensure_ascii=False),
                            ),
                        )
                    )

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            # Extract content before first tool call
            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content
            )

        except Exception:
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_name(self, name_str: str) -> str:
        """Extract name from quoted string."""
        name_str = name_str.strip()
        if (
            name_str.startswith('"')
            and name_str.endswith('"')
            or name_str.startswith("'")
            and name_str.endswith("'")
        ):
            return name_str[1:-1]
        return name_str

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct type."""
        if value.lower() == "null":
            return None

        param_type = param_type.lower()
        if param_type in ["string", "str", "text"]:
            return value
        elif param_type in ["integer", "int"]:
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type in ["number", "float"]:
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (ValueError, TypeError):
                return value
        elif param_type in ["boolean", "bool"]:
            return value.lower() in ["true", "1"]
        elif param_type in ["object", "array"]:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        else:
            # Try JSON parse first, fallback to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

    def _convert_params_with_schema(
        self,
        function_name: str,
        param_dict: dict[str, str],
        request: ChatCompletionRequest | None,
    ) -> dict[str, Any]:
        """Convert raw string param values using the tool schema types."""
        param_config: dict = {}
        if request and request.tools:
            for tool in request.tools:
                if (
                    hasattr(tool, "function")
                    and tool.function.name == function_name
                    and hasattr(tool.function, "parameters")
                ):
                    schema = tool.function.parameters
                    if isinstance(schema, dict) and "properties" in schema:
                        param_config = schema["properties"]
                    break

        converted: dict[str, Any] = {}
        for name, value in param_dict.items():
            param_type = "string"
            if name in param_config and isinstance(param_config[name], dict):
                param_type = param_config[name].get("type", "string")
            converted[name] = self._convert_param_value(value, param_type)
        return converted

    def _extract_delta_tool_calls(
        self,
        current_text: str,
        request: ChatCompletionRequest | None,
    ) -> list[DeltaToolCall]:
        """Extract DeltaToolCalls from newly completed <invoke> blocks.

        Tracks progress via ``current_tool_index`` so each block is
        extracted exactly once across successive streaming calls.
        """
        complete_invokes = self.invoke_complete_regex.findall(current_text)
        delta_tool_calls: list[DeltaToolCall] = []

        while len(complete_invokes) > self.current_tool_index:
            invoke_name, invoke_body = complete_invokes[self.current_tool_index]
            param_dict = self._parse_invoke_params(invoke_body)
            if param_dict is None:
                self.current_tool_index += 1
                continue

            converted = self._convert_params_with_schema(
                invoke_name, param_dict, request
            )
            args_json = json.dumps(converted, ensure_ascii=False)
            idx = self.current_tool_index
            self.current_tool_index += 1

            self.prev_tool_call_arr.append(
                {"name": invoke_name, "arguments": converted}
            )
            self.streamed_args_for_tool.append(args_json)

            delta_tool_calls.append(
                DeltaToolCall(
                    index=idx,
                    id=self._generate_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=invoke_name,
                        arguments=args_json,
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
        previous_token_ids: Sequence[int],  # pylint: disable=unused-argument
        current_token_ids: Sequence[int],  # pylint: disable=unused-argument
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming model output.

        Uses a buffer-until-complete-invoke strategy: tokens are buffered
        until a complete invoke block is available, then parsed and emitted
        in one shot.
        """

        # Reset state on new request (parser is reused across streams).
        tool_call_starting = self.tool_call_start_token in delta_text
        if not previous_text or tool_call_starting:
            self._reset_streaming_state()
            self.is_tool_call_started = tool_call_starting

        # Pass through content before any tool call.
        if not self.is_tool_call_started:
            return DeltaMessage(content=delta_text) if delta_text else None

        # Capture content before the start token.
        content_before = None
        if tool_call_starting:
            before = delta_text[: delta_text.index(self.tool_call_start_token)]
            content_before = before or None

        # Extract newly completed <invoke> blocks as DeltaToolCalls.
        delta_tool_calls = self._extract_delta_tool_calls(current_text, request)

        if delta_tool_calls or content_before:
            return DeltaMessage(
                content=content_before,
                tool_calls=delta_tool_calls,
            )

        # Special tokens (EOS, </function_calls>) arrive with no decoded
        # text. Return non-None so the serving framework reaches the
        # finish-reason handling path instead of skipping.
        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")

        return None
