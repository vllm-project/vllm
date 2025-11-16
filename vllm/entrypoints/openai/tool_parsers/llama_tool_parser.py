# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import partial_json_parser
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

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
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.entrypoints.openai.tool_parsers.utils import (
    find_common_prefix,
    is_complete_json,
    partial_json_loads,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class Llama3JsonToolParser(ToolParser):
    """
    Tool call parser for Llama 3.x and 4 models intended for use with the
    examples/tool_chat_template_llama.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser llama3_json or
    llama4_json are set.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list
        self.bot_token = "<|python_tag|>"
        self.bot_token_id = tokenizer.encode(self.bot_token, add_special_tokens=False)[
            0
        ]

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        Only extracts JSON content and ignores any surrounding plain text.
        Supports both single JSON and multiple JSONs separated by semicolons.
        """
        # Quick check before running regex
        if not (self.bot_token in model_output or "{" in model_output):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            # Find the start of JSON object
            start_idx = model_output.find("{")
            if start_idx == -1:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            # Use iterative parsing with brace counting to extract JSON objects
            # This handles strings with braces correctly by tracking string boundaries
            json_objects = self._extract_json_objects(model_output[start_idx:])

            if not json_objects:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls: list[ToolCall] = []
            for json_str in json_objects:
                obj = json.loads(json_str)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=obj["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(
                                obj["arguments"]
                                if "arguments" in obj
                                else obj["parameters"],
                                ensure_ascii=False,
                            ),
                        ),
                    )
                )

            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=None
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # return information to just treat the tool call as regular JSON
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_json_objects(self, text: str) -> list[str]:
        """
        Extract JSON objects from text using brace counting with string awareness.
        Handles nested JSON and strings containing braces correctly.
        Supports multiple JSONs separated by semicolons.
        """
        json_objects = []
        i = 0

        while i < len(text):
            # Skip whitespace and semicolons
            while i < len(text) and text[i] in " \t\n\r;":
                i += 1

            if i >= len(text) or text[i] != "{":
                break

            # Track braces and string state
            brace_count = 0
            in_string = False
            escape_next = False
            start = i

            while i < len(text):
                char = text[i]

                if escape_next:
                    escape_next = False
                    i += 1
                    continue

                if char == "\\":
                    escape_next = True
                    i += 1
                    continue

                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif char == "{" and not in_string:
                    brace_count += 1
                elif char == "}" and not in_string:
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON object
                        json_objects.append(text[start : i + 1])
                        i += 1
                        break

                i += 1

            # If we didn't find a complete object, stop
            if brace_count != 0:
                break

        return json_objects

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
        if not (
            current_text.startswith(self.bot_token) or current_text.startswith("{")
        ):
            return DeltaMessage(content=delta_text)

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                # depending on the prompt format the Llama model may or may not
                # prefix the output with the <|python_tag|> token
                start_idx = (
                    len(self.bot_token)
                    if current_text.startswith(self.bot_token)
                    else 0
                )
                while start_idx < len(current_text):
                    (obj, end_idx) = partial_json_loads(current_text[start_idx:], flags)
                    is_complete.append(
                        is_complete_json(current_text[start_idx : start_idx + end_idx])
                    )
                    start_idx += end_idx + len("; ")
                    # depending on the prompt Llama can use
                    # either arguments or parameters
                    if "parameters" in obj:
                        assert "arguments" not in obj, (
                            "model generated both parameters and arguments"
                        )
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)
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
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
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
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=make_tool_call_id(),
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                else:
                    delta = None

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                delta = None

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
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
