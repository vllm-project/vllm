# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

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
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)

logger = init_logger(__name__)

ESCAPED_ARGUMENTS_PARAM_NAME = "__vllm_param_arguments__"


class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """
    DeepSeek V4 DSML tool parser.

    V4 keeps the V3.2 DSML invoke/parameter grammar, but wraps tool calls in
    ``<｜DSML｜tool_calls>`` instead of ``<｜DSML｜function_calls>``.
    """

    tool_call_start_token: str = "<｜DSML｜tool_calls>"
    tool_call_end_token: str = "</｜DSML｜tool_calls>"

    def get_structural_tag(self, request: ChatCompletionRequest):
        return get_model_structural_tag(
            model="deepseek_v4",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )

    @staticmethod
    def _function_name(tool) -> str | None:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                return function.get("name")
            return getattr(function, "name", None)
        return getattr(getattr(tool, "function", None), "name", None)

    @staticmethod
    def _function_parameters(tool):
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                return function.get("parameters")
            return getattr(function, "parameters", None)
        return getattr(getattr(tool, "function", None), "parameters", None)

    def _extract_param_name(self, param_name: str) -> str:
        if param_name == ESCAPED_ARGUMENTS_PARAM_NAME:
            return "arguments"
        return param_name

    def _get_param_config(
        self,
        request: ChatCompletionRequest | None,
        function_name: str | None,
    ) -> dict[str, dict]:
        if not request or not request.tools or not function_name:
            return {}

        for tool in request.tools:
            if self._function_name(tool) != function_name:
                continue
            params = self._function_parameters(tool)
            if isinstance(params, dict):
                properties = params.get("properties")
                if isinstance(properties, dict):
                    return properties
            return {}

        return {}

    def _coerce_param_value(
        self,
        value: str,
        *,
        string_attr: str,
        param_type,
    ):
        if string_attr == "true":
            return value
        if param_type:
            return self._convert_param_value(value, param_type)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _repair_param_dict(
        param_dict: dict,
        param_config: dict[str, dict],
    ) -> dict:
        allowed = set(param_config.keys())
        for wrapper in ("arguments", "input"):
            if set(param_dict.keys()) != {wrapper} or wrapper in allowed:
                continue
            inner = param_dict[wrapper]
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    return param_dict
            if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
                return inner
        return param_dict

    def _parse_invoke_params(
        self,
        invoke_str: str,
        request: ChatCompletionRequest | None = None,
        function_name: str | None = None,
    ) -> dict:
        param_config = self._get_param_config(request, function_name)
        param_dict = {}

        for param_name, string_attr, param_val in self.parameter_complete_regex.findall(
            invoke_str
        ):
            original_param_name = param_name
            param_name = self._extract_param_name(param_name)
            param_type = None
            if (
                original_param_name == ESCAPED_ARGUMENTS_PARAM_NAME
                and "arguments" in param_config
            ):
                param_type = param_config["arguments"].get("type")
            elif param_name in param_config and isinstance(
                param_config[param_name], dict
            ):
                param_type = param_config[param_name].get("type")

            param_dict[param_name] = self._coerce_param_value(
                param_val,
                string_attr=string_attr,
                param_type=param_type,
            )

        return self._repair_param_dict(param_dict, param_config)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract DeepSeek V4 DSML tool calls from complete model output."""
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = []
            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                for invoke_name, invoke_content in self.invoke_complete_regex.findall(
                    tool_call_match
                ):
                    param_dict = self._parse_invoke_params(
                        invoke_content,
                        request=request,
                        function_name=invoke_name,
                    )
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

            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content
            )

        except Exception:
            logger.exception("Error extracting DeepSeek V4 tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_delta_tool_calls(
        self,
        current_text: str,
        request: ChatCompletionRequest | None,
    ) -> list[DeltaToolCall]:
        complete_invokes = self.invoke_complete_regex.findall(current_text)
        delta_tool_calls: list[DeltaToolCall] = []

        while len(complete_invokes) > self.current_tool_index:
            invoke_name, invoke_body = complete_invokes[self.current_tool_index]
            param_dict = self._parse_invoke_params(
                invoke_body,
                request=request,
                function_name=invoke_name,
            )
            args_json = json.dumps(param_dict, ensure_ascii=False)
            idx = self.current_tool_index
            self.current_tool_index += 1

            self.prev_tool_call_arr.append(
                {"name": invoke_name, "arguments": param_dict}
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
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not previous_text:
            self._reset_streaming_state()

        content = self._extract_content(current_text)
        delta_tool_calls = self._extract_delta_tool_calls(current_text, request)

        if (
            not delta_text
            and self.tool_call_start_token not in current_text
            and self._sent_content_idx < len(current_text)
        ):
            held_content = current_text[self._sent_content_idx :]
            self._sent_content_idx = len(current_text)
            content = (content or "") + held_content

        if delta_tool_calls or content:
            return DeltaMessage(content=content, tool_calls=delta_tool_calls)

        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")

        return None
