# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Sequence
from typing import Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("hermes")
class Hermes2ProToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if isinstance(self.model_tokenizer, MistralTokenizer):
            logger.error(
                "Detected Mistral tokenizer when using a Hermes model")
            self.model_tokenizer = self.model_tokenizer.tokenizer

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = [
        ]  # map what has been streamed for each tool so far to a list

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
        self.scratch_pad_regex = re.compile(
            r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL)

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
                "Hermes 2 Pro Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        # sanity check; avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        else:

            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = (
                    self.tool_call_regex.findall(model_output))

                # load the JSON, and then use it to build the Function and
                # Tool Call
                raw_function_calls = [
                    json.loads(match[0] if match[0] else match[1])
                    for match in function_call_tuples
                ]
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(function_call["arguments"],
                                                 ensure_ascii=False)))
                    for function_call in raw_function_calls
                ]

                content = model_output[:model_output.
                                       find(self.tool_call_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None)

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

        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        # check to see if we should be streaming a tool call - is there a
        if self.tool_call_start_token_id not in current_token_ids:
            logger.debug("No tool call tokens found!")
            return DeltaMessage(content=delta_text)

        try:

            # figure out where we are in the parsing by counting tool call
            # start & end tags
            prev_tool_start_count = previous_token_ids.count(
                self.tool_call_start_token_id)
            prev_tool_end_count = previous_token_ids.count(
                self.tool_call_end_token_id)
            cur_tool_start_count = current_token_ids.count(
                self.tool_call_start_token_id)
            cur_tool_end_count = current_token_ids.count(
                self.tool_call_end_token_id)
            tool_call_portion = None
            text_portion = None

            # case: if we're generating text, OR rounding out a tool call
            if (cur_tool_start_count == cur_tool_end_count
                    and prev_tool_end_count == cur_tool_end_count
                    and self.tool_call_end_token not in delta_text):
                logger.debug("Generating text content! skipping tool parsing.")
                return DeltaMessage(content=delta_text)

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text + delta_text
                tool_call_portion = full_text.split(
                    self.tool_call_start_token)[-1].split(
                        self.tool_call_end_token)[0].rstrip()
                delta_text = delta_text.split(
                    self.tool_call_end_token)[0].rstrip()
                text_portion = delta_text.split(
                    self.tool_call_end_token)[-1].lstrip()

            # case: if tool open & close tag counts don't match, we're doing
            # imaginary "else" block here
            # something with tools with this diff.
            # flags for partial JSON parting. exported constants from
            # "Allow" are handled via BIT MASK
            flags = Allow.ALL if self.current_tool_name_sent \
                else Allow.ALL & ~Allow.STR

            # case -- we're starting a new tool call
            if (cur_tool_start_count > cur_tool_end_count
                    and cur_tool_start_count > prev_tool_start_count):
                if len(delta_token_ids) > 1:
                    tool_call_portion = current_text.split(
                        self.tool_call_start_token)[-1]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None

                # set cursors and state appropriately
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # case -- we're updating an existing tool call
            elif (cur_tool_start_count > cur_tool_end_count
                  and cur_tool_start_count == prev_tool_start_count):

                # get the portion of the text that's the tool call
                tool_call_portion = current_text.split(
                    self.tool_call_start_token)[-1]
                text_portion = None

            # case -- the current tool call is being closed.
            elif (cur_tool_start_count == cur_tool_end_count
                  and cur_tool_end_count >= prev_tool_end_count):
                if (self.prev_tool_call_arr is None
                        or len(self.prev_tool_call_arr) == 0):
                    logger.debug(
                        "attempting to close tool call, but no tool call")
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments")
                if diff:
                    diff = diff.encode('utf-8').decode(
                        'unicode_escape') if diff is str else diff
                    if ('"}' not in delta_text):
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    logger.debug(
                        "Finishing tool and found diff that had not "
                        "been streamed yet: %s", diff)
                    self.streamed_args_for_tool[self.current_tool_id] \
                        += diff
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      function=DeltaFunctionCall(
                                          arguments=diff).model_dump(
                                              exclude_none=True))
                    ])

            # case -- otherwise we're just generating text
            else:
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                delta = DeltaMessage(tool_calls=[], content=text)
                return delta

            try:

                current_tool_call = partial_json_parser.loads(
                    tool_call_portion or "{}",
                    flags) if tool_call_portion else None
                logger.debug("Parsed tool call %s", current_tool_call)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug('not enough tokens to parse into JSON yet')
                return None
            except json.decoder.JSONDecodeError:
                logger.debug("unable to parse JSON")
                return None

            # case - we haven't sent the tool name yet. If it's available, send
            #   it. otherwise, wait until it's available.
            if not self.current_tool_name_sent:
                if (current_tool_call is None):
                    return None
                function_name: Union[str, None] = current_tool_call.get("name")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      type="function",
                                      id=f"chatcmpl-tool-{random_uuid()}",
                                      function=DeltaFunctionCall(
                                          name=function_name).model_dump(
                                              exclude_none=True))
                    ])
                else:
                    return None
            # case -- otherwise, send the tool call delta

            # if the tool call portion is None, send the delta as text
            if tool_call_portion is None:
                # if there's text but not tool calls, send that -
                # otherwise None to skip chunk
                delta = DeltaMessage(content=delta_text) \
                    if text_portion is not None else None
                return delta

            # now, the nitty-gritty of tool calls
            # now we have the portion to parse as tool call.

            logger.debug("Trying to parse current tool call with ID %s",
                         self.current_tool_id)

            # if we're starting a new tool call, push an empty object in as
            #   a placeholder for the arguments
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # main logic for tool parsing here - compare prev. partially-parsed
            #   JSON to the current partially-parsed JSON
            prev_arguments = (
                self.prev_tool_call_arr[self.current_tool_id].get("arguments"))
            cur_arguments = current_tool_call.get("arguments")

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            # case -- no arguments have been created yet. skip sending a delta.
            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None

            # case -- prev arguments are defined, but non are now.
            #   probably impossible, but not a fatal error - just keep going
            elif not cur_arguments and prev_arguments:
                logger.error("should be impossible to have arguments reset "
                             "mid-call. skipping streaming anything.")
                delta = None

            # case -- we now have the first info about arguments available from
            #   autocompleting the JSON
            elif cur_arguments and not prev_arguments:

                cur_arguments_json = json.dumps(cur_arguments,
                                                ensure_ascii=False)
                logger.debug("finding %s in %s", delta_text,
                             cur_arguments_json)

                # get the location where previous args differ from current
                if (delta_text not in cur_arguments_json[:-2]):
                    return None
                args_delta_start_loc = cur_arguments_json[:-2]. \
                                           rindex(delta_text) + \
                                           len(delta_text)

                # use that to find the actual delta
                arguments_delta = cur_arguments_json[:args_delta_start_loc]
                logger.debug("First tokens in arguments received: %s",
                             arguments_delta)

                delta = DeltaMessage(tool_calls=[
                    DeltaToolCall(index=self.current_tool_id,
                                  function=DeltaFunctionCall(
                                      arguments=arguments_delta).model_dump(
                                          exclude_none=True))
                ])
                self.streamed_args_for_tool[self.current_tool_id] \
                    += arguments_delta

            # last case -- we have an update to existing arguments.
            elif cur_arguments and prev_arguments:
                if isinstance(delta_text, str) and len(delta_text.rstrip(
                )) > 1 and delta_text.rstrip()[-1] == '}':
                    delta_text = delta_text.rstrip()[:-1]

                logger.debug("got diff %s", delta_text)

                delta = DeltaMessage(tool_calls=[
                    DeltaToolCall(index=self.current_tool_id,
                                  function=DeltaFunctionCall(
                                      arguments=delta_text).model_dump(
                                          exclude_none=True))
                ])
                self.streamed_args_for_tool[self.current_tool_id] \
                    += delta_text

            # handle saving the state for the current tool into
            # the "prev" list for use in diffing for the next iteration
            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = \
                    current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.
