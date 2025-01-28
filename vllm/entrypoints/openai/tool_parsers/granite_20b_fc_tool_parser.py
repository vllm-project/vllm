import json
import re
from json import JSONDecoder
from typing import Dict, Sequence, Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.utils import (consume_space,
                                                        find_common_prefix,
                                                        is_complete_json,
                                                        partial_json_loads)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("granite-20b-fc")
class Granite20bFCToolParser(ToolParser):
    """
    Tool call parser for the granite-20b-functioncalling model intended
    for use with the examples/tool_chat_template_granite20b_fc.jinja
    template.

    Used when --enable-auto-tool-choice --tool-call-parser granite-20-fc
    are all set
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.bot_token = "<function_call>"
        self.tool_start_token = self.bot_token
        self.tool_call_regex = re.compile(r"<function_call>\s*")

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        if self.tool_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        dec = JSONDecoder()
        try:
            matches = list(self.tool_call_regex.finditer(model_output))
            logger.debug("Found %d tool call matches", len(matches))

            raw_function_calls = []

            for i, match in enumerate(matches):
                # position after the <function_call> tag
                start_of_json = match.end()
                # end_index == the start of the next function call
                # (if exists)
                next_function_call_start = (matches[i + 1].start() if i +
                                            1 < len(matches) else None)

                raw_function_calls.append(
                    dec.raw_decode(
                        model_output[start_of_json:next_function_call_start])
                    [0])

            logger.debug("Extracted %d tool calls", len(raw_function_calls))
            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(function_call["arguments"]),
                    ),
                ) for function_call in raw_function_calls
            ]

            content = model_output[:model_output.find(self.bot_token)]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception as e:
            logger.error("Error in extracting tool call from response %s", e)
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

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

        if len(current_text) < len(
                self.bot_token) and self.bot_token.startswith(current_text):
            return None

        if not current_text.startswith(self.bot_token):
            return DeltaMessage(content=delta_text)

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent \
            else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                start_idx = len(self.bot_token)
                start_idx = consume_space(start_idx, current_text)

                while start_idx < len(current_text):
                    (obj,
                     end_idx) = partial_json_loads(current_text[start_idx:],
                                                   flags)
                    is_complete.append(
                        is_complete_json(current_text[start_idx:start_idx +
                                                      end_idx]))
                    start_idx += end_idx
                    start_idx = consume_space(start_idx, current_text)
                    start_idx += len(self.bot_token)
                    start_idx = consume_space(start_idx, current_text)
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug('not enough tokens to parse into JSON yet')
                return None

            # select as the current tool call the one we're on the state at
            current_tool_call: Dict = tool_call_arr[self.current_tool_id] \
                if len(tool_call_arr) > 0 else {}

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return None

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (len(tool_call_arr) > 0
                  and len(tool_call_arr) > self.current_tool_id + 1):

                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(
                            self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        logger.debug("got arguments diff: %s", argument_diff)
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id,
                                          function=DeltaFunctionCall(
                                              arguments=argument_diff).
                                          model_dump(exclude_none=True))
                        ])
                        self.streamed_args_for_tool[
                            self.current_tool_id] += argument_diff
                    else:
                        delta = None
                else:
                    delta = None
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:

                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      type="function",
                                      id=f"chatcmpl-tool-{random_uuid()}",
                                      function=DeltaFunctionCall(
                                          name=function_name).model_dump(
                                              exclude_none=True))
                    ])
                    self.current_tool_name_sent = True
                else:
                    delta = None

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                delta = None

                if cur_arguments:
                    sent = len(
                        self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = self.prev_tool_call_arr[
                        self.current_tool_id].get("arguments")

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:

                            prefix = find_common_prefix(
                                prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id,
                                          function=DeltaFunctionCall(
                                              arguments=argument_diff).
                                          model_dump(exclude_none=True))
                        ])
                        self.streamed_args_for_tool[
                            self.current_tool_id] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception as e:
            logger.error("Error trying to handle streaming tool call: %s", e)
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction "
                "error")
            return None
