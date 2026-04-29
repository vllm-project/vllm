# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from xgrammar import StructuralTag, get_model_structural_tag
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)

class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """
    DeepSeek V4 DSML tool parser.

    V4 keeps the V3.2 DSML invoke/parameter grammar, but wraps tool calls in
    ``<｜DSML｜tool_calls>`` instead of ``<｜DSML｜function_calls>``.
    """

    tool_call_start_token: str = "<｜DSML｜tool_calls>"
    tool_call_end_token: str = "</｜DSML｜tool_calls>"
    
    def get_structural_tag(self, request: ChatCompletionRequest) -> StructuralTag:
        def _tool_to_dict(tool: ChatCompletionToolsParam | dict) -> dict:
            if isinstance(tool, dict):
                return tool
            if hasattr(tool, "model_dump"):
                return tool.model_dump()
            if hasattr(tool, "dict"):
                return tool.dict()
            raise TypeError(f"Unsupported tool type: {type(tool)}")

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            converted_tool_choice = request.tool_choice.model_dump()
            converted_tools = []
            for tool in request.tools:
                tool_dict = _tool_to_dict(tool)
                tool_name = tool_dict.get("function", {}).get("name")
                if tool_name == request.tool_choice.function.name:
                    converted_tools.append(tool_dict)
        else:
            converted_tool_choice = request.tool_choice
            converted_tools = [_tool_to_dict(tool) for tool in request.tools]

        return get_model_structural_tag(
            model="deepseek_v4",
            tools=converted_tools,
            tool_choice=converted_tool_choice,
            reasoning=request.include_reasoning,
        ) 
