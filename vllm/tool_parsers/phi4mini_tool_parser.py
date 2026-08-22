# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any

import regex as re
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

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
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    find_common_prefix,
    partial_json_loads,
)

logger = init_logger(__name__)


def _partial_bot_token_len(text: str, bot_token: str) -> int:
    """Length of the longest suffix of ``text`` that starts ``bot_token``.

    The ``functools`` marker is not a special token, so it is split across
    several ordinary tokens while streaming. Any trailing text that could still
    grow into the marker is held back instead of being emitted as content.
    """
    max_len = min(len(text), len(bot_token) - 1)
    for length in range(max_len, 0, -1):
        if bot_token.startswith(text[-length:]):
            return length
    return 0


class Phi4MiniJsonToolParser(ToolParser):
    """
    Tool call parser for phi-4-mini models intended for use with the
    examples/tool_chat_template_llama.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser phi4_mini_json
    are all set
    """

    json_decoder: json.JSONDecoder = json.JSONDecoder()

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list
        self.bot_token: str = "functools"
        # number of leading characters already streamed back as content
        self.streamed_content_len: int = 0

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        logger.debug("Model output: %s", model_output)

        pattern = r"functools\[(.*?)\]"
        matches = re.search(pattern, model_output, re.DOTALL)

        if not matches:
            logger.debug("No function calls found")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_call_arr: list[dict[str, Any]] = []
            try:
                json_content = "[" + matches.group(1) + "]"

                function_call_arr = json.loads(json_content)
                logger.debug(
                    "Successfully extracted %d function calls", len(function_call_arr)
                )
            except json.JSONDecodeError as e:
                logger.error(
                    "Failed to parse function calls from model output. Error: %s",
                    str(e),
                )

            tool_calls: list[ToolCall] = [
                ToolCall(
                    id=make_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(
                            raw_function_call["arguments"]
                            if "arguments" in raw_function_call
                            else raw_function_call["parameters"],
                            ensure_ascii=False,
                        ),
                    ),
                )
                for raw_function_call in function_call_arr
            ]

            # get any content before the tool call
            ret = ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=None
            )
            return ret

        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _stream_content(self, current_text: str, up_to: int) -> DeltaMessage | None:
        """Emit the not-yet-streamed content in ``current_text[:up_to]``."""
        if up_to <= self.streamed_content_len:
            return None
        content = current_text[self.streamed_content_len : up_to]
        self.streamed_content_len = up_to
        return DeltaMessage(content=content)

    def _parse_partial_tool_calls(
        self, tool_calls_text: str, flags: Allow
    ) -> tuple[list[dict[str, Any]], list[bool]]:
        """Parse the (possibly incomplete) ``[...]`` array after the marker.

        Returns the tool call objects decoded so far along with, for each of
        them, whether its JSON object was already terminated.
        """
        tool_call_arr: list[dict[str, Any]] = []
        is_complete: list[bool] = []

        start_idx = 1 if tool_calls_text.startswith("[") else 0
        while start_idx < len(tool_calls_text):
            while (
                start_idx < len(tool_calls_text)
                and tool_calls_text[start_idx] in " \t\r\n,"
            ):
                start_idx += 1
            if start_idx >= len(tool_calls_text) or tool_calls_text[start_idx] == "]":
                break

            remainder = tool_calls_text[start_idx:]
            try:
                # A terminated object is decoded directly; this also keeps the
                # array's closing bracket away from the partial JSON parser,
                # which cannot cope with an unmatched closing bracket.
                obj, end_idx = self.json_decoder.raw_decode(remainder)
                complete = True
            except json.JSONDecodeError:
                try:
                    obj, end_idx = partial_json_loads(remainder, flags)
                except (MalformedJSON, ValueError, IndexError):
                    # Not enough tokens to parse into JSON yet, or the model
                    # emitted something that is not a tool call at all.
                    break
                complete = False

            if end_idx <= 0:
                break

            # phi-4-mini templates emit either "arguments" or "parameters"
            if "parameters" in obj and "arguments" not in obj:
                obj["arguments"] = obj["parameters"]

            tool_call_arr.append(obj)
            is_complete.append(complete)
            start_idx += end_idx

        return tool_call_arr, is_complete

    def _argument_diff(
        self, tool_call: dict[str, Any], is_complete: bool
    ) -> str | None:
        """Arguments of ``tool_call`` that have not been streamed yet."""
        cur_arguments = tool_call.get("arguments")
        if not cur_arguments:
            return None

        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
        sent = len(self.streamed_args_for_tool[self.current_tool_id])

        if is_complete:
            argument_diff = cur_args_json[sent:]
        else:
            prev_arguments = None
            if self.current_tool_id < len(self.prev_tool_call_arr):
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )
            if not prev_arguments:
                return None
            prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
            if cur_args_json == prev_args_json:
                return None
            # Only stream the part both parses agree on, so that closing
            # quotes and braces are not sent prematurely.
            argument_diff = find_common_prefix(prev_args_json, cur_args_json)[sent:]

        if not argument_diff:
            return None

        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
        return argument_diff

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
        if not previous_text:
            # a new response is starting, drop any state left from the last one
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []
            self.streamed_content_len = 0

        bot_index = current_text.find(self.bot_token)

        # No tool call started yet: stream everything as content, holding back
        # any trailing text that could still grow into the marker.
        if bot_index == -1:
            safe_end = len(current_text) - _partial_bot_token_len(
                current_text, self.bot_token
            )
            return self._stream_content(current_text, safe_end)

        # Any text before the marker is regular content and must be flushed
        # before the tool call deltas are emitted.
        content_delta = self._stream_content(current_text, bot_index)
        if content_delta is not None:
            return content_delta

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending an incomplete string since OpenAI only
        # ever allows sending the entire tool/function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        tool_calls_text = current_text[bot_index + len(self.bot_token) :]

        try:
            tool_call_arr, is_complete = self._parse_partial_tool_calls(
                tool_calls_text, flags
            )

            # case: nothing but the opening bracket has been streamed yet
            if len(tool_call_arr) == 0:
                return None

            # case: we are starting a new tool in the array. Any arguments of
            # the previous tool that were completed by the same chunk must be
            # flushed before the cursor moves on.
            if len(tool_call_arr) > self.current_tool_id + 1:
                pending = None
                if self.current_tool_id >= 0 and self.current_tool_name_sent:
                    pending = self._argument_diff(
                        tool_call_arr[self.current_tool_id], is_complete=True
                    )

                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)

                if pending is not None:
                    self.prev_tool_call_arr = tool_call_arr
                    return pending

            current_tool_call: dict = tool_call_arr[self.current_tool_id]

            function_name = None
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if not function_name:
                    self.prev_tool_call_arr = tool_call_arr
                    return None
                self.current_tool_name_sent = True

            # The name and the first arguments chunk may be emitted together,
            # but only once the object is terminated: while it is still partial
            # the arguments were parsed with string values suppressed, so the
            # diff would not be a prefix of the final arguments.
            argument_diff = None
            if function_name is None or is_complete[self.current_tool_id]:
                argument_diff = self._argument_diff(
                    current_tool_call, is_complete[self.current_tool_id]
                )

            self.prev_tool_call_arr = tool_call_arr

            if function_name is None and argument_diff is None:
                return None

            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function" if function_name else None,
                        id=make_tool_call_id() if function_name else None,
                        function=DeltaFunctionCall(
                            name=function_name,
                            arguments=argument_diff,
                        ).model_dump(exclude_none=True),
                    )
                ]
            )

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None
