# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

import vllm.envs as envs
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
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.tool_parsers.utils import (
    find_common_prefix,
    is_complete_json,
    partial_json_loads,
)

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
        # Simple regex to find opening braces - we'll use JSON decoder for parsing
        # This handles arbitrary nesting depth correctly
        self.tool_call_start_regex = re.compile(r"\{")
        self.json_decoder = json.JSONDecoder()

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

        # Keep track of the end index of the last parsed JSON object
        # so we don't parse inner brackets
        end_index = -1
        tool_calls: list[ToolCall] = []

        try:
            for match in self.tool_call_start_regex.finditer(
                model_output, timeout=envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
            ):
                start_index = match.start()
                # Skip if this brace is inside a previously parsed JSON object
                if start_index <= end_index:
                    continue

                try:
                    obj, json_end_index = self.json_decoder.raw_decode(
                        model_output[start_index:]
                    )
                    end_index = start_index + json_end_index

                    # raise KeyError if missing
                    name = obj["name"]
                    arguments_or_params = (
                        obj["arguments"] if "arguments" in obj else obj["parameters"]
                    )

                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=name,
                                # function call args are JSON but as a string
                                arguments=json.dumps(
                                    arguments_or_params, ensure_ascii=False
                                ),
                            ),
                        )
                    )
                except KeyError as e:
                    # Missing required key
                    missing_key = str(e).strip("'\"")
                    logger.exception(
                        "Couldn't extract tool call from JSON response. "
                        "Required key '%s' not present. "
                        "Returning output in content with empty tool calls.",
                        missing_key,
                    )
                    return ExtractedToolCallInformation(
                        tools_called=False, tool_calls=[], content=model_output
                    )
                except Exception:
                    # Any other error during parsing
                    logger.exception(
                        "Error in extracting tool call from response. "
                        "Returning output in content with empty tool calls"
                    )
                    return ExtractedToolCallInformation(
                        tools_called=False, tool_calls=[], content=model_output
                    )
        except TimeoutError:
            logger.warning("Regex timeout occurred when matching tool call pattern.")
            logger.debug(
                "Regex timeout occurred when matching user input: %s", model_output
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # If we have valid tool calls, return them normally
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=None
            )

        # No valid tool calls found
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
