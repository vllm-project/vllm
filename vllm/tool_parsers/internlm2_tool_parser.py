# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import find_common_prefix, is_complete_json

logger = init_logger(__name__)


class Internlm2ToolParser(ToolParser):
    # A tool call is a JSON object wrapped in these markers:
    #   <|action_start|><|plugin|>{...}<|action_end|>
    _ACTION_START = "<|action_start|><|plugin|>"
    _ACTION_END = "<|action_end|>"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.position = 0

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because internlm use the special
            # tokens to indicate the start and end of the tool calls
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
    ) -> DeltaMessage | None:
        if "<|action_start|>" not in current_text:
            self.position = len(current_text)
            return DeltaMessage(content=delta_text)

        marker_start = current_text.find(self._ACTION_START, self.position)
        if marker_start == -1:
            # The marker has not arrived in full yet.
            return None

        # Text between the previous block and this one is content.
        content = current_text[self.position : marker_start] or None
        self.position = marker_start

        try:
            tool_calls = self._stream_blocks(current_text, marker_start, delta_text)
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return DeltaMessage(content=content) if content else None

        if not content and not tool_calls:
            return None
        return DeltaMessage(content=content, tool_calls=tool_calls or None)

    def _stream_blocks(
        self, current_text: str, marker_start: int, delta_text: str
    ) -> list[DeltaToolCall]:
        """Stream the tool call block starting at *marker_start* and every
        block that follows it.

        InternLM2 wraps each tool call in its own markers, so an output may
        hold more than one block. The next block is picked up as soon as the
        current one is closed.
        """
        tool_calls: list[DeltaToolCall] = []
        while True:
            action_start = marker_start + len(self._ACTION_START)
            action, block_end = self._split_action(current_text, action_start)
            tool_call = self._stream_action(action, delta_text)
            if tool_call is not None:
                tool_calls.append(tool_call)

            if block_end == -1:
                # The block is still being generated; wait for more text.
                break

            next_marker = current_text.find(self._ACTION_START, block_end)
            # The next block starts a new tool call, so its name still has to
            # be sent.
            self.current_tool_name_sent = False
            if next_marker == -1:
                self.position = block_end
                break
            self.position = next_marker
            marker_start = next_marker
        return tool_calls

    def _split_action(self, current_text: str, action_start: int) -> tuple[str, int]:
        """Return the action of the current block and the offset just past its
        end marker, or -1 when the block has not been closed yet."""
        end = current_text.find(self._ACTION_END, action_start)
        action = current_text[action_start : end if end != -1 else None]
        return action.strip(), (end + len(self._ACTION_END)) if end != -1 else -1

    def _stream_action(self, action: str, delta_text: str) -> DeltaToolCall | None:
        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        parsable_arr = action
        try:
            tool_call_arr: dict = partial_json_parser.loads(parsable_arr, flags)
        except partial_json_parser.core.exceptions.MalformedJSON:
            logger.debug("not enough tokens to parse into JSON yet")
            return None

        # if the current tool name hasn't been sent, send if available
        # - otherwise send nothing
        if not self.current_tool_name_sent:
            function_name = tool_call_arr.get("name")
            if not function_name:
                return None
            self.current_tool_id = self.current_tool_id + 1
            self.current_tool_name_sent = True
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append(tool_call_arr)

            function_call = DeltaFunctionCall(name=function_name)
            # A block that arrives whole carries its arguments with it, and no
            # further delta may arrive to stream them in.
            cur_arguments = self.get_arguments(tool_call_arr)
            if cur_arguments is not None and is_complete_json(parsable_arr):
                arguments = json.dumps(cur_arguments, ensure_ascii=False)
                function_call.arguments = arguments
                self.streamed_args_for_tool[self.current_tool_id] = arguments

            tool_call_arr["arguments"] = cur_arguments
            return DeltaToolCall(
                index=self.current_tool_id,
                type="function",
                id=make_tool_call_id(),
                function=function_call.model_dump(exclude_none=True),
            )

        # now we know we're on the same tool call and we're streaming
        # arguments
        prev_arguments = self.get_arguments(
            self.prev_tool_call_arr[self.current_tool_id]
        )
        cur_arguments = self.get_arguments(tool_call_arr)

        delta: DeltaToolCall | None
        # not arguments generated
        if not cur_arguments and not prev_arguments:
            delta = None
        # will never happen
        elif not cur_arguments and prev_arguments:
            logger.error("INVARIANT - impossible to have arguments reset mid-arguments")
            delta = None
        # first time to get parameters
        elif cur_arguments and not prev_arguments:
            cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)

            match_start = cur_arguments_json.find(delta_text)
            if match_start != -1:
                arguments_delta = cur_arguments_json[: match_start + len(delta_text)]
            elif is_complete_json(parsable_arr):
                # Complete in this delta: send whole, don't drop.
                arguments_delta = cur_arguments_json
            else:
                # Still partial: wait for more text.
                return None
            delta = DeltaToolCall(
                index=self.current_tool_id,
                function=DeltaFunctionCall(arguments=arguments_delta).model_dump(
                    exclude_none=True
                ),
            )
            self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
        # both prev and cur parameters, send the increase parameters
        else:
            cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
            prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
            sent = len(self.streamed_args_for_tool[self.current_tool_id])

            # Partial JSON is serialized with synthetic closers, so
            # only the part shared with the previous parse is final.
            if is_complete_json(parsable_arr):
                argument_diff = cur_args_json[sent:]
            elif cur_args_json != prev_args_json:
                prefix = find_common_prefix(prev_args_json, cur_args_json)
                argument_diff = prefix[sent:]
            else:
                argument_diff = ""

            delta = DeltaToolCall(
                index=self.current_tool_id,
                function=DeltaFunctionCall(arguments=argument_diff).model_dump(
                    exclude_none=True
                ),
            )
            self.streamed_args_for_tool[self.current_tool_id] += argument_diff

        # check to see if the name is defined and has been sent. if so,
        # stream the name - otherwise keep waiting
        # finish by setting old and returning None as base case
        tool_call_arr["arguments"] = self.get_arguments(tool_call_arr)
        self.prev_tool_call_arr[self.current_tool_id] = tool_call_arr
        return delta

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        text = model_output
        if self._ACTION_START in text:
            # Each `<|action_start|><|plugin|>...<|action_end|>` block is one
            # tool call. Text before the first block is the assistant content.
            segments = text.split(self._ACTION_START)
            content = segments[0]

            tool_calls: list[ToolCall] = []
            for action in segments[1:]:
                action = action.split(self._ACTION_END)[0]
                action = action[action.find("{") :]
                action_dict = json.loads(action)
                name, parameters = (
                    action_dict["name"],
                    json.dumps(
                        action_dict.get("parameters", action_dict.get("arguments", {})),
                        ensure_ascii=False,
                    ),
                )
                tool_calls.append(
                    ToolCall(function=FunctionCall(name=name, arguments=parameters))
                )

            if tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if len(content) > 0 else None,
                )

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=text
        )
