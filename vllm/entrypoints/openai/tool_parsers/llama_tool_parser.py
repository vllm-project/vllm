# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Sequence
from json import JSONDecoder
from typing import Union

import partial_json_parser
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.utils import (find_common_prefix,
                                                        is_complete_json,
                                                        partial_json_loads)
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("llama3_json")
class Llama3JsonToolParser(ToolParser):
    """
    Tool call parser for Llama 3.1 models intended for use with the
    examples/tool_chat_template_llama.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser llama3_json 
    are all set
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = [
        ]  # map what has been streamed for each tool so far to a list
        self.bot_token = "<|python_tag|>"
        self.bot_token_id = tokenizer.encode(self.bot_token,
                                             add_special_tokens=False)[0]
        self.tool_call_regex = re.compile(r"\[{.*?}\]", re.DOTALL)

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        # case -- if a tool call token is not present, return a text response
        if not (model_output.startswith(self.bot_token)
                or model_output.startswith('{')):
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            # load the JSON, and then use it to build the Function and
            # Tool Call
            dec = JSONDecoder()
            function_call_arr = []

            # depending on the prompt format the Llama model may or may not
            # prefix the output with the <|python_tag|> token
            start_idx = len(self.bot_token) if model_output.startswith(
                self.bot_token) else 0
            while start_idx < len(model_output):
                (obj, end_idx) = dec.raw_decode(model_output[start_idx:])
                start_idx += end_idx + len('; ')
                function_call_arr.append(obj)

            tool_calls: list[ToolCall] = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(raw_function_call["arguments"] \
                                if "arguments" in raw_function_call \
                                else raw_function_call["parameters"])))
                for raw_function_call in function_call_arr
            ]

            # get any content before  the tool call
            ret = ExtractedToolCallInformation(tools_called=True,
                                               tool_calls=tool_calls,
                                               content=None)
            return ret

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # return information to just treat the tool call as regular JSON
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

        if not (current_text.startswith(self.bot_token)
                or current_text.startswith('{')):
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
                # depending on the prompt format the Llama model may or may not
                # prefix the output with the <|python_tag|> token
                start_idx = len(self.bot_token) if current_text.startswith(
                    self.bot_token) else 0
                while start_idx < len(current_text):
                    (obj,
                     end_idx) = partial_json_loads(current_text[start_idx:],
                                                   flags)
                    is_complete.append(
                        is_complete_json(current_text[start_idx:start_idx +
                                                      end_idx]))
                    start_idx += end_idx + len('; ')
                    # depending on the prompt Llama can use
                    # either arguments or parameters
                    if "parameters" in obj:
                        assert "arguments" not in obj, \
                            "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug('not enough tokens to parse into JSON yet')
                return None

            # select as the current tool call the one we're on the state at
            current_tool_call: dict = tool_call_arr[self.current_tool_id] \
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

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction "
                "error")
            return None
