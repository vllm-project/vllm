# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import extract_intermediate_diff
from vllm.utils import random_uuid

logger = init_logger(__name__)


class StepAudio2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.position = 0
        self.previous_end_position = 0  # 跟踪上一个函数调用结束的位置

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because internlm use the special
            # tokens to indicated the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        return request

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
        if "<tool_call>" not in current_text:
            self.position = len(current_text)
            return DeltaMessage(content=delta_text)

        # 处理函数调用结束和新的函数调用开始
        if "</tool_call>" in current_text[self.previous_end_position :]:
            end_pos = current_text.find("</tool_call>", self.previous_end_position)
            next_start = current_text.find("<tool_call>", end_pos)

            if next_start > end_pos and next_start > self.position:
                self.previous_end_position = end_pos + len("</tool_call>")
                self.position = next_start
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({})
            elif end_pos > self.position:
                if delta_text and (
                    "<tool_call>" in delta_text or "</tool_call>" in delta_text
                ):
                    return None
                return DeltaMessage(content=delta_text)

        if "<tool_call>" not in current_text[self.position :]:
            if delta_text and (
                "<tool_call>" in delta_text or "</tool_call>" in delta_text
            ):
                return None
            return DeltaMessage(content=delta_text)

        new_delta = current_text[self.position :]
        text, action = new_delta.split("<tool_call>", 1)

        if len(text) > 0:
            self.position = self.position + len(text)
            return DeltaMessage(content=text)

        action = action.strip()
        if "</tool_call>" in action:
            action = action.split("</tool_call>")[0].strip()

        # 解析新格式：function\nforecast_weather\n{"location": "shanghai"}
        action_parts = action.split("\n", 2)
        if len(action_parts) < 3:
            return None

        function_type, function_name, arguments = action_parts

        try:
            tool_call_arr = {"name": function_name}

            try:
                if arguments:
                    args_dict = json.loads(arguments)
                    tool_call_arr["parameters"] = args_dict
            except json.JSONDecodeError:
                logger.debug("Failed to parse arguments as JSON")
                tool_call_arr["parameters"] = None
                if self.current_tool_name_sent:
                    return None

            # 如果这是第一个工具调用，初始化 current_tool_id
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.streamed_args_for_tool = [""]
                self.prev_tool_call_arr = [{}]
                self.current_tool_name_sent = False

            # 确保 prev_tool_call_arr 有足够的元素
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # 确保 streamed_args_for_tool 有足够的元素
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            if not self.current_tool_name_sent:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=f"chatcmpl-tool-{random_uuid()}",
                            function=DeltaFunctionCall(name=function_name).model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                )
                self.current_tool_name_sent = True
                self.streamed_args_for_tool[self.current_tool_id] = ""
            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                    "parameters"
                )
                cur_arguments = tool_call_arr.get("parameters")

                # not arguments generated
                if not cur_arguments and not prev_arguments:
                    delta = None
                # will never happen
                elif not cur_arguments and prev_arguments:
                    logger.error(
                        "INVARIANT - impossible to have arguments reset mid-arguments"
                    )
                    delta = None
                # first time to get parameters
                elif cur_arguments and not prev_arguments:
                    # 直接使用完整的参数而不是尝试查找 delta_text
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)

                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=cur_arguments_json
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = (
                        cur_arguments_json
                    )
                # both prev and cur parameters, send the increase parameters
                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)

                    argument_diff = extract_intermediate_diff(
                        cur_args_json, prev_args_json
                    )

                    if not argument_diff:
                        delta = None
                    else:
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=argument_diff
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += (
                            argument_diff
                        )

            # 更新当前工具调用的信息
            self.prev_tool_call_arr[self.current_tool_id] = tool_call_arr
            return delta
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        text = model_output
        tools = request.tools

        # 检查是否包含函数调用
        if "<tool_call>" in text:
            # 首先分离文本和工具调用部分
            parts = text.split("<tool_call>", 1)
            content_text = parts[0]
            remaining_text = parts[1]

            tool_calls = []
            remaining_part = ""
            # 处理所有的函数调用
            while "<tool_call>" in remaining_text or remaining_text:
                # 如果有结束标记，则提取当前函数调用
                if "</tool_call>" in remaining_text:
                    action_part, remaining_part = remaining_text.split(
                        "</tool_call>", 1
                    )

                    # 解析函数调用信息
                    action_parts = action_part.split("\n", 2)
                    if len(action_parts) >= 3:
                        function_type, function_name, parameters = action_parts

                        try:
                            params_dict = json.loads(parameters)
                            params_str = json.dumps(params_dict, ensure_ascii=False)

                            # 验证函数名是否在可用工具列表中
                            if not tools or function_name in [
                                t.function.name for t in tools
                            ]:
                                tool_calls.append(
                                    ToolCall(
                                        function=FunctionCall(
                                            name=function_name, arguments=params_str
                                        )
                                    )
                                )
                        except json.JSONDecodeError:
                            # 参数解析失败，跳过当前函数调用
                            pass

                    # 检查是否有更多函数调用
                    if "<tool_call>" in remaining_part:
                        _, remaining_text = remaining_part.split("<tool_call>", 1)
                    else:
                        break
                else:
                    # 没有结束标记，可能是不完整的函数调用
                    break

            content_text += remaining_part

            # 如果找到了有效的工具调用，则返回结果
            if tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content_text if content_text else None,
                )

        # 没有有效的工具调用，返回原始文本
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=text
        )


if __name__ == "__main__":
    parser = StepAudio2ToolParser(tokenizer=None)
    request = ChatCompletionRequest(
        model="step_audio_2",
        messages=[{"role": "user", "content": "你好"}],
    )
    stream_output = [
        "xxx",
        "yyy",
        "<tool_call>",
        "function\n",
        "forecast",
        "_weather\n",
        '{"location":',
        ' "shanghai"}',
        "</tool_call>",
        "<tool_call>",
        "function\n",
        "forecast",
        "_temperature\n",
        '{"location": ',
        '"beijing"}',
        "</tool_call>",
        "zzz",
        "",
    ]
    non_stream_output = "".join(stream_output)
    print("****************")
    print(parser.extract_tool_calls(non_stream_output, request))
    print("****************")
    current_text = ""
    for delta_text in stream_output:
        current_text += delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        print(delta)
