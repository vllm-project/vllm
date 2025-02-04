# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Dict, List, Sequence, Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizers import MistralTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("jamba")
class JambaToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if isinstance(self.model_tokenizer, MistralTokenizer):
            raise ValueError(
                "Detected a MistralTokenizer tokenizer when using a Jamba model"
            )

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: List[str] = [
        ]  # map what has been streamed for each tool so far to a list

        self.tool_calls_start_token: str = "<tool_calls>"
        self.tool_calls_end_token: str = "</tool_calls>"

        self.tool_calls_regex = re.compile(
            rf"{self.tool_calls_start_token}(.*?){self.tool_calls_end_token}",
            re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")
        self.tool_calls_start_token_id = self.vocab.get(
            self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(
            self.tool_calls_end_token)
        if (self.tool_calls_start_token_id is None
                or self.tool_calls_end_token_id is None):
            raise RuntimeError(
                "Jamba Tool parser could not locate tool calls start/end "
                "tokens in the tokenizer!")

    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because jamba use the special
            # tokens to indicate the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:

        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        else:

            try:
                # use a regex to find the tool call between the tags
                function_calls = self.tool_calls_regex.findall(model_output)[0]

                # load the JSON, and then use it to build the Function and
                # Tool Call
                raw_function_calls = json.loads(function_calls)
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(function_call["arguments"])))
                    for function_call in raw_function_calls
                ]

                content = model_output[:model_output.
                                       find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if
                    (len(content) > 0 and content != " ") else None)

            except Exception:
                logger.exception(
                    "Error in extracting tool call from response.")
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

        # if the tool call token is not in the tokens generated so far, append
        # output to contents since it's not a tool
        if self.tool_calls_start_token not in current_text:
            return DeltaMessage(content=delta_text)

        # if the tool call token ID IS in the tokens generated so far, that
        # means we're parsing as tool calls now

        # handle if we detected the start of tool calls token which means
        # the start of tool calling
        if (self.tool_calls_start_token_id in delta_token_ids
                and len(delta_token_ids) == 1):
            # if it's the only token, return None, so we don't send a chat
            # completion and don't send a control token
            return None

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent \
            else Allow.ALL & ~Allow.STR
        try:

            # Extract the tool calls between the special tool call tokens
            parsable_arr = current_text.split(
                self.tool_calls_start_token)[-1].split(
                    self.tool_calls_end_token)[0]

            # tool calls are generated in an array, so do partial JSON
            # parsing on the entire array
            try:
                tool_call_arr: List[Dict] = partial_json_parser.loads(
                    parsable_arr, flags)
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
                    diff: Union[str, None] = current_tool_call.get("arguments")

                    if diff:
                        diff = json.dumps(diff).replace(
                            self.streamed_args_for_tool[self.current_tool_id],
                            "")
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id,
                                          function=DeltaFunctionCall(
                                              arguments=diff).model_dump(
                                                  exclude_none=True))
                        ])
                        self.streamed_args_for_tool[
                            self.current_tool_id] += diff
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

            # case: update an existing tool - this is handled below

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            if not self.current_tool_name_sent:
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

                prev_arguments = self.prev_tool_call_arr[
                    self.current_tool_id].get("arguments")
                cur_arguments = current_tool_call.get("arguments")

                new_text = delta_text.replace("\'", "\"")

                if not cur_arguments and not prev_arguments:

                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error(
                        "INVARIANT - impossible to have arguments reset "
                        "mid-arguments")
                    delta = None
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments)
                    logger.debug("finding %s in %s", new_text,
                                 cur_arguments_json)

                    arguments_delta = cur_arguments_json[:cur_arguments_json.
                                                         index(new_text) +
                                                         len(new_text)]
                    logger.debug("First tokens in arguments received: %s",
                                 arguments_delta)
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      function=DeltaFunctionCall(
                                          arguments=arguments_delta).
                                      model_dump(exclude_none=True))
                    ])
                    self.streamed_args_for_tool[
                        self.current_tool_id] += arguments_delta

                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments)
                    prev_args_json = json.dumps(prev_arguments)
                    logger.debug("Searching for diff between \n%s\n%s",
                                 cur_args_json, prev_args_json)

                    argument_diff = extract_intermediate_diff(
                        cur_args_json, prev_args_json)
                    logger.debug("got arguments diff: %s", argument_diff)
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      function=DeltaFunctionCall(
                                          arguments=argument_diff).model_dump(
                                              exclude_none=True))
                    ])
                    self.streamed_args_for_tool[
                        self.current_tool_id] += argument_diff
                else:
                    # try parsing it with regular JSON - if it works we're
                    # at the end, and we need to send the difference between
                    # tokens streamed so far and the valid JSON
                    delta = None

            # check to see if the name is defined and has been sent. if so,
            # stream the name - otherwise keep waiting
            # finish by setting old and returning None as base case
            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction "
                "error")
            return None
