# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Union

import partial_json_parser
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


@ToolParserManager.register_module(["internlm"])
class Internlm2ToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.position = 0

    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because internlm use the special
            # tokens to indicated the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        return request

    def get_arguments(self, obj):
        if "parameters" in obj:
            return obj.get("parameters")
        elif "arguments" in obj:
            return obj.get("arguments")
        return None

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
        if '<|action_start|>' not in current_text:
            self.position = len(current_text)
            return DeltaMessage(content=delta_text)
        # if the tool call is sended, return a empty delta message
        # to make sure the finish_reason will be send correctly.
        if self.current_tool_id > 0:
            return DeltaMessage(content='')

        last_pos = self.position
        if '<|action_start|><|plugin|>' not in current_text[last_pos:]:
            return None

        new_delta = current_text[last_pos:]
        text, action = new_delta.split('<|action_start|><|plugin|>')

        if len(text) > 0:
            self.position = self.position + len(text)
            return DeltaMessage(content=text)

        action = action.strip()
        action = action.split('<|action_end|>'.strip())[0]

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent \
            else Allow.ALL & ~Allow.STR

        try:
            parsable_arr = action

            # tool calls are generated in an object in inernlm2
            # it's not support parallel tool calls
            try:
                tool_call_arr: dict = partial_json_parser.loads(
                    parsable_arr, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug('not enough tokens to parse into JSON yet')
                return None

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            if not self.current_tool_name_sent:
                function_name = tool_call_arr.get("name")
                if function_name:
                    self.current_tool_id = self.current_tool_id + 1
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      type="function",
                                      id=random_tool_call_id(),
                                      function=DeltaFunctionCall(
                                          name=function_name).model_dump(
                                              exclude_none=True))
                    ])
                    self.current_tool_name_sent = True
                    self.streamed_args_for_tool.append("")
                else:
                    delta = None
            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                prev_arguments = self.get_arguments(
                    self.prev_tool_call_arr[self.current_tool_id])
                cur_arguments = self.get_arguments(tool_call_arr)

                # not arguments generated
                if not cur_arguments and not prev_arguments:
                    delta = None
                # will never happen
                elif not cur_arguments and prev_arguments:
                    logger.error(
                        "INVARIANT - impossible to have arguments reset "
                        "mid-arguments")
                    delta = None
                # first time to get parameters
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments,
                                                    ensure_ascii=False)

                    arguments_delta = cur_arguments_json[:cur_arguments_json.
                                                         index(delta_text) +
                                                         len(delta_text)]
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      function=DeltaFunctionCall(
                                          arguments=arguments_delta).
                                      model_dump(exclude_none=True))
                    ])
                    self.streamed_args_for_tool[
                        self.current_tool_id] += arguments_delta
                # both prev and cur parameters, send the increase parameters
                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments,
                                               ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments,
                                                ensure_ascii=False)

                    argument_diff = extract_intermediate_diff(
                        cur_args_json, prev_args_json)

                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      function=DeltaFunctionCall(
                                          arguments=argument_diff).model_dump(
                                              exclude_none=True))
                    ])
                    self.streamed_args_for_tool[
                        self.current_tool_id] += argument_diff

            # check to see if the name is defined and has been sent. if so,
            # stream the name - otherwise keep waiting
            # finish by setting old and returning None as base case
            tool_call_arr["arguments"] = self.get_arguments(tool_call_arr)
            self.prev_tool_call_arr = [tool_call_arr]
            return delta
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction "
                "error")
            return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        text = model_output
        tools = request.tools
        if '<|action_start|><|plugin|>' in text:
            text, action = text.split('<|action_start|><|plugin|>')
            action = action.split('<|action_end|>'.strip())[0]
            action = action[action.find('{'):]
            action_dict = json.loads(action)
            name, parameters = action_dict['name'], json.dumps(
                action_dict.get('parameters', action_dict.get('arguments',
                                                              {})),
                ensure_ascii=False)

            if not tools or name not in [t.function.name for t in tools]:
                ExtractedToolCallInformation(tools_called=False,
                                             tool_calls=[],
                                             content=text)

            tool_calls = [
                ToolCall(
                    function=FunctionCall(name=name, arguments=parameters))
            ]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=text if len(text) > 0 else None)

        return ExtractedToolCallInformation(tools_called=False,
                                            tool_calls=[],
                                            content=text)
