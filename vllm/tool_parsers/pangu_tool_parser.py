# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
from collections.abc import Sequence

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
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
from vllm.tool_parsers.utils import (
    find_common_prefix,
    is_complete_json,
)

logger = init_logger(__name__)


class PanguToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = (
            "<|tool_call_start|>"
            if self.vocab.get("<|tool_call_start|>")
            else "[unused11]"
        )
        self.tool_call_end_token: str = (
            "<|tool_call_end|>" if self.vocab.get("<|tool_call_end|>") else "[unused12]"
        )
        self.pattern = (
            re.escape(self.tool_call_start_token)
            + "(.*?)"
            + re.escape(self.tool_call_end_token)
        )
        self.tool_call_regex = re.compile(self.pattern, re.DOTALL)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Pangu Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )
        self.is_json_complete = False
        self.text_after_start_token = ""

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool call from a complete model response.
        """
        # case -- if a tool call token is not present, return a text response
        if not (
            self.tool_call_start_token in model_output
            and self.tool_call_end_token in model_output
        ):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            raw_function_calls = []
            # use a regex to find the tool call between the tags
            function_call_tuples = self.tool_call_regex.findall(model_output)

            # load the JSON, and then use it to build the Function and
            # Tool Call
            for function_call_str in function_call_tuples:
                function_call = json.loads(function_call_str)
                raw_function_calls.extend(function_call)

            tool_calls: list[ToolCall] = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(
                            function_call["arguments"], ensure_ascii=False
                        ),
                    ),
                )
                for function_call in raw_function_calls
            ]
            content = model_output[: model_output.find(self.tool_call_start_token)]

            ret = ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

            return ret

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # return information to just treat the tool call as regular JSON
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

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
        if self.tool_call_end_token_id in delta_token_ids and len(delta_token_ids) == 1:
            # if it's the only token, return None, so we don't send a chat
            # completion and don't send a control token
            return None

        if (
            self.tool_call_start_token_id in delta_token_ids
            and len(delta_token_ids) == 1
        ):
            # if it's the only token, return None, so we don't send a chat
            # completion and don't send a control token
            return None

        if (
            self.tool_call_end_token in current_text
            and self.tool_call_end_token not in delta_text
        ):
            return None

        if self.tool_call_start_token not in current_text:
            return DeltaMessage(content=delta_text)

        if self.tool_call_start_token in delta_text:
            texts = delta_text.split(self.tool_call_start_token)
            text_before_start_token = texts[0]
            if text_before_start_token:
                return DeltaMessage(content=text_before_start_token)

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_portion = current_text.split(self.tool_call_start_token)[
                -1
            ].split(self.tool_call_end_token)[0]
            try:
                tool_call_arr: list[dict] = partial_json_parser.loads(
                    tool_call_portion, flags
                )

                self.is_json_complete = is_complete_json(tool_call_portion)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            # select as the current tool call the one we're on the state at
            current_tool_call: dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return None

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (
                len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1
            ):
                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                delta = self.start_new_tool(tool_call_arr, current_tool_call)
                return delta
            # case: update an existing tool - this is handled below

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            elif not self.current_tool_name_sent:
                delta = self.send_current_tool_name_or_none(current_tool_call)
            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                delta = self.stream_arguments(current_tool_call)

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None

    def start_new_tool(self, tool_call_arr, current_tool_call):
        if self.current_tool_id >= 0:
            cur_arguments = current_tool_call.get("arguments")
            if cur_arguments is not None:
                cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                sent = len(self.streamed_args_for_tool[self.current_tool_id])
                argument_diff = cur_args_json[sent:]

                logger.debug("got arguments diff: %s", argument_diff)
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
                self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            else:
                delta = None
        else:
            delta = None
        # re-set stuff pertaining to progress in the current tool
        self.current_tool_id += 1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool.append("")
        self.is_json_complete = False
        logger.debug("starting on new tool %d", self.current_tool_id)

        return delta

    def send_current_tool_name_or_none(self, current_tool_call):
        function_name = current_tool_call.get("name")
        cur_arguments = current_tool_call.get("arguments")
        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
        argument_diff = None
        if (
            self.is_json_complete
            and cur_args_json is not None
            and not self.streamed_args_for_tool[-1]
        ):
            argument_diff = cur_args_json

        if function_name and argument_diff is None:
            delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(name=function_name).model_dump(
                            exclude_none=True
                        ),
                    )
                ]
            )
            self.current_tool_name_sent = True
        elif function_name and argument_diff is not None:
            delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=function_name, arguments=argument_diff
                        ).model_dump(exclude_none=True),
                    )
                ]
            )
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff
            self.current_tool_name_sent = True
            self.is_json_complete = False
        else:
            delta = None
        return delta

    def stream_arguments(self, current_tool_call):
        cur_arguments = current_tool_call.get("arguments")
        delta = None
        if (
            self.is_json_complete
            and not cur_arguments
            and not self.streamed_args_for_tool[-1]
        ):
            argument_diff = "{}"
            delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=argument_diff).model_dump(
                            exclude_none=True
                        ),
                    )
                ]
            )
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff

        if cur_arguments:
            sent = len(self.streamed_args_for_tool[self.current_tool_id])
            cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )

            argument_diff = None
            if self.is_json_complete:
                argument_diff = cur_args_json[sent:]
            elif prev_arguments:
                prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                if cur_args_json != prev_args_json:
                    prefix = find_common_prefix(prev_args_json, cur_args_json)
                    argument_diff = prefix[sent:]

            if argument_diff is not None:
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
                self.streamed_args_for_tool[self.current_tool_id] += argument_diff

        return delta
