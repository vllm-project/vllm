# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
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
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import (
    find_common_prefix,
    is_complete_json,
    partial_json_loads,
)

logger = init_logger(__name__)


class OpenPanguV2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        self.tool_call_start_token: str = "<|tool_call_start|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
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
                "Pangu Tool parser could not locate tool calls start/end "
                "tokens in the tokenizer!"
            )
        self.text_after_start_token = ""

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        # case -- if a tool call token is not present, return a text response
        if not (
            self.tool_call_start_token in model_output
            and model_output.find(self.tool_call_end_token) != -1
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
                            function_call["arguments"]
                            if "arguments" in function_call
                            else function_call["parameters"],
                            ensure_ascii=False,
                        ),
                    ),
                )
                for function_call in raw_function_calls
            ]
            content = model_output[: model_output.find(self.tool_call_start_token)]

            # get any content before  the tool call
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

        if (
            self.tool_call_start_token_id in delta_token_ids
            and len(delta_token_ids) == 1
        ):
            # if it's the only token, return None, so we don't send a chat
            # completion and don't send a control token
            return None

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            self.is_complete = []
            current_text = current_text.split(self.tool_call_start_token)[-1].split(
                self.tool_call_end_token
            )[0]
            current_text = current_text[
                current_text.find("[") + 1 : current_text.rfind("]")
            ]
            start_idx = current_text.find("{")

            try:
                while start_idx < len(current_text):
                    (obj, end_idx) = partial_json_loads(current_text[start_idx:], flags)
                    current_tool_text = current_text[start_idx : start_idx + end_idx]
                    next_tool_text = current_text[start_idx + end_idx :]
                    if next_tool_text.find("{") != -1:
                        next_tool_start_idx = next_tool_text.find("{")
                        next_tool_start = next_tool_text[:next_tool_start_idx]
                    else:
                        next_tool_start = next_tool_text
                    self.is_complete.append(is_complete_json(current_tool_text))
                    start_idx += end_idx + len(next_tool_start)

                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return None

            # select as the current tool call the one we're on the state at
            current_tool_call: dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            if len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
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
                        self.streamed_args_for_tool[self.current_tool_id] += (
                            argument_diff
                        )
                    else:
                        delta = None
                else:
                    delta = None
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    cur_arguments = current_tool_call.get("arguments")
                    cur_args_json = None
                    if (
                        cur_arguments is not None
                        and self.is_complete[self.current_tool_id]
                    ):
                        # If args are already present in this chunk, emit them
                        # together with tool name to avoid empty-argument tool call.
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=make_tool_call_id(),
                                function=DeltaFunctionCall(
                                    name=function_name, arguments=cur_args_json
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    if cur_args_json is not None:
                        self.streamed_args_for_tool[self.current_tool_id] = (
                            cur_args_json
                        )
                    self.current_tool_name_sent = True
                else:
                    delta = None

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                delta = None
                if cur_arguments is not None:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )

                    argument_diff = None
                    if self.is_complete[self.current_tool_id]:
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
                        self.streamed_args_for_tool[self.current_tool_id] += (
                            argument_diff
                        )

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None
