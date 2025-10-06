# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from enum import Enum, auto
from random import choices
from string import ascii_letters, digits
from typing import Optional, Union

import ijson
import regex as re
from pydantic import Field

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer

logger = init_logger(__name__)

ALPHANUMERIC = ascii_letters + digits


class StreamingState(Enum):
    """Enum for tracking the current streaming parsing state."""
    WAITING_FOR_TOOL_START = auto()
    WAITING_FOR_TOOL_KEY = auto(
    )  # waiting for the "name" or "arguments" key to be complete
    PARSING_NAME = auto()
    PARSING_NAME_COMPLETED = auto()
    WAITING_FOR_ARGUMENTS_START = auto()
    PARSING_ARGUMENTS = auto()
    PARSING_ARGUMENTS_COMPLETED = auto()
    TOOL_COMPLETE = auto()
    ALL_TOOLS_COMPLETE = auto()


class MistralToolCall(ToolCall):
    id: str = Field(
        default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id():
        # Mistral Tool Call Ids must be alphanumeric with a length of 9.
        # https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L299
        return "".join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:
        return id.isalnum() and len(id) == 9


def _is_pre_v11_tokeniser(model_tokenizer: AnyTokenizer) -> bool:
    return not (isinstance(model_tokenizer, MistralTokenizer) \
        and model_tokenizer.version >= 11)


@ToolParserManager.register_module("mistral")
class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral 7B Instruct v0.3, intended for use with
    - [`mistral_common`](https://github.com/mistralai/mistral-common/)
    - the examples/tool_chat_template_mistral.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser mistral are all set
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if not isinstance(self.model_tokenizer, MistralTokenizer):
            logger.info("Non-Mistral tokenizer detected when using a Mistral "
                        "model...")

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.current_tool_id: int = -1
        self.streaming_state: StreamingState = \
            StreamingState.WAITING_FOR_TOOL_START

        # For streaming pre v11 tokenizer tool calls
        self.current_tool_name: Optional[str] = None
        self.current_tool_mistral_id: Optional[str] = None
        self.starting_new_tool = False
        if _is_pre_v11_tokeniser(self.model_tokenizer):
            self.parse_coro = ijson.parse_coro(
                self.update_stream_state_pre_v11_tokenizer())

        self.bot_token = "[TOOL_CALLS]"
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        if not _is_pre_v11_tokeniser(self.model_tokenizer):
            self.fn_name_regex = re.compile(
                r'([a-zA-Z0-9_-]+)(\{[\s\S]*?\})(?=\s*$|,|\s)', re.DOTALL)
        else:
            self.fn_name_regex = None

        if self.bot_token_id is None:
            raise RuntimeError(
                "Mistral Tool Parser could not locate the tool call token in "
                "the tokenizer!")

    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if not isinstance(
                self.model_tokenizer, MistralTokenizer
        ) and request.tools and request.tool_choice != 'none':
            # Do not skip special tokens when using chat template
            # with Mistral parser as TOOL_CALL token is needed
            # for tool detection.
            # Note: we don't want skip_special_tokens=False
            # with MistralTokenizer as it is incompatible
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response. Requires
        find-and-replacing single quotes with double quotes for JSON parsing,
        make sure your tool call arguments don't ever include quotes!
        """

        # case -- if a tool call token is not present, return a text response
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        # first remove the BOT token
        tool_content = model_output.replace(self.bot_token, "").strip()

        try:
            # we first try to directly load the json as parsing very nested
            # jsons is difficult
            try:
                if self.fn_name_regex:
                    function_call_arr = []
                    for single_tool_content in model_output.split(
                            self.bot_token):
                        matches = self.fn_name_regex.findall(
                            single_tool_content)

                        for match in matches:
                            fn_name = match[0]
                            args = match[1]

                            # fn_name is encoded outside serialized json dump
                            # only arguments are serialized
                            function_call_arr.append({
                                "name":
                                fn_name,
                                "arguments":
                                json.loads(args)
                            })
                else:
                    function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                # use a regex to find the part corresponding to the tool call.
                # NOTE: This use case should not happen if the model is trained
                # correctly. It's an easy possible fix so it's included, but
                # can be brittle for very complex / highly nested tool calls
                raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
                function_call_arr = json.loads(raw_tool_call)

            # Tool Call
            tool_calls: list[MistralToolCall] = [
                MistralToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(raw_function_call["arguments"],
                                             ensure_ascii=False)))
                for raw_function_call in function_call_arr
            ]

            # get any content before  the tool call
            content = model_output.split(self.bot_token)[0]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if len(content) > 0 else None)

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # return information to just treat the tool call as regular JSON
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=tool_content)

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

        if self.bot_token not in current_text:
            # if the tool call token is not in the tokens generated so far,
            # append output to contents since it's not a tool
            return DeltaMessage(content=delta_text)

        # if the tool call token IS in the tokens generated so far, that
        # means we're parsing as tool calls now

        if _is_pre_v11_tokeniser(self.model_tokenizer):
            return self._extract_tool_calls_streaming_pre_v11_tokenizer(
                delta_text=delta_text, )
        else:
            return self._extract_tool_calls_streaming(delta_text=delta_text)

    def _extract_tool_calls_streaming(
        self,
        delta_text: str,
    ) -> Union[DeltaMessage, None]:
        """
        Extracts tool calls for Mistral models
        doing tool calls of the following format:
        `[TOOL_CALLS]add{"a": 3.5, "b": 4}`
        """
        additional_content: str = ""
        if self.streaming_state == StreamingState.WAITING_FOR_TOOL_START:
            # this is the first tool call
            assert self.bot_token in delta_text
            if not delta_text.startswith(self.bot_token):
                additional_content += delta_text.split(self.bot_token)[0]
                delta_text = self.bot_token + "".join(
                    delta_text.split(self.bot_token)[1:])

        delta_tool_calls = self._generate_delta_tool_call(delta_text)
        if not additional_content and len(delta_tool_calls) == 0:
            if self.streaming_state in [
                    StreamingState.PARSING_ARGUMENTS,
                    StreamingState.PARSING_ARGUMENTS_COMPLETED,
                    StreamingState.TOOL_COMPLETE,
                    StreamingState.ALL_TOOLS_COMPLETE
            ]:
                # Return an empty DeltaMessage once the tool calls are all done
                # so that finish_reason gets set.
                return DeltaMessage()
            else:
                # return None when the tool is not likely to be finished
                # This can occur when the name is being parsed for example
                # and we wait for the name to be complete
                # before sendingthe function name
                return None

        delta = DeltaMessage()
        if additional_content:
            delta.content = additional_content
        if len(delta_tool_calls) > 0:
            delta.tool_calls = delta_tool_calls

        # HACK: serving_chat.py inspects the internal state of tool parsers
        # when determining it's final streaming delta, automatically
        # adding autocompleted JSON.
        # These two lines avoid that nonsense while ensuring finish_reason
        # is set to tool_calls when at least one tool is called.
        if delta_tool_calls and not self.prev_tool_call_arr:
            self.prev_tool_call_arr = [{"arguments": {}}]
        return delta

    def _generate_delta_tool_call(self,
                                  delta_text: str) -> list[DeltaToolCall]:
        if delta_text == "" or delta_text is None:
            return []
        delta_function_name = None
        tool_id = None
        if self.streaming_state not in [
                StreamingState.PARSING_NAME, StreamingState.PARSING_ARGUMENTS
        ] and delta_text.startswith(self.bot_token):
            self.current_tool_id += 1
            self.streaming_state = StreamingState.PARSING_NAME
            delta_text = delta_text.replace(self.bot_token, "", 1)
        if self.streaming_state == StreamingState.PARSING_NAME:
            if self.current_tool_name is None:
                self.current_tool_name = ""
            # The name stops where the arguments start
            # And the arguments start with the `{` char
            if "{" in delta_text:
                tool_id = MistralToolCall.generate_random_id()
                delta_function_name = delta_text.split("{")[0]
                self.current_tool_name += delta_function_name
                delta_text = delta_text[len(delta_function_name):]
                self.streaming_state = StreamingState.PARSING_ARGUMENTS
            else:
                # we want to send the tool name once it's complete
                self.current_tool_name += delta_text
                return []
        if self.streaming_state == StreamingState.PARSING_ARGUMENTS:
            next_function_text = None
            if self.bot_token in delta_text:
                # current tool call is over
                delta_arguments = ""
                delta_arguments += delta_text.split(self.bot_token)[0]
                next_function_text = delta_text[len(delta_arguments):]
                self.streaming_state = StreamingState.TOOL_COMPLETE
            else:
                delta_arguments = delta_text
            ret = []
            if self.current_tool_name or delta_arguments:
                ret += [
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=tool_id,
                        function=DeltaFunctionCall(
                            name=self.current_tool_name,
                            arguments=delta_arguments).model_dump(
                                exclude_none=True),
                    )
                ]
                self.current_tool_name = None
            if next_function_text:
                ret += self._generate_delta_tool_call(next_function_text)
            return ret
        # Should not happen
        return []

    @ijson.coroutine
    def update_stream_state_pre_v11_tokenizer(self):
        while True:
            (prefix, event, value) = (yield)

            if prefix == "item" and event == "start_map":
                self.streaming_state = StreamingState.WAITING_FOR_TOOL_KEY
            if prefix == "item" and event == "map_key" and value == "name":
                self.streaming_state = StreamingState.PARSING_NAME
            if prefix == "item.name" and event == "string":
                self.current_tool_name = value
                self.streaming_state = StreamingState.PARSING_NAME_COMPLETED
            if prefix == "item"\
                and event == "map_key" and value == "arguments":
                self.streaming_state =\
                    StreamingState.WAITING_FOR_ARGUMENTS_START
            if prefix == "item.arguments" and event == "start_map":
                self.streaming_state = StreamingState.PARSING_ARGUMENTS
            if prefix == "item.arguments" and event == "end_map":
                self.streaming_state = \
                    StreamingState.PARSING_ARGUMENTS_COMPLETED
            if prefix == "item" and event == "end_map":
                self.streaming_state = StreamingState.TOOL_COMPLETE
            if prefix == "" and event == "end_array":
                self.streaming_state = StreamingState.ALL_TOOLS_COMPLETE

    def _extract_tool_calls_streaming_pre_v11_tokenizer(
            self, delta_text: str) -> Union[DeltaMessage, None]:
        """
        Extracts tool calls for Mistral models
        doing tool calls of the following format:
        `[TOOL_CALLS][{"name": "add", "arguments":{"a": 3.5, "b": 4}}`
        """
        assert self.parse_coro is not None
        content = None
        delta_tool_calls: list[DeltaToolCall] = []
        current_tool_call: DeltaToolCall = DeltaToolCall(
            index=self.current_tool_id, type="function")
        current_tool_call_modified = False
        if self.bot_token in delta_text:
            # this is the first tool call
            if not delta_text.startswith(self.bot_token):
                content = delta_text.split(self.bot_token)[0]
            delta_text = "".join(delta_text.split(self.bot_token)[1:])

        # Cut smartly the delta text to catch the ijson events
        # as ijson does not give us the index in the text at each event.
        # We need to cut so that we know
        # where in the text the events are emited from.
        while len(delta_text) > 0:
            streaming_state_before_parse = self.streaming_state

            if self.streaming_state == StreamingState.WAITING_FOR_TOOL_START:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_opening_curly_braces=1,
                )
            elif self.streaming_state == StreamingState.WAITING_FOR_TOOL_KEY:
                # Wait until another key is sent
                # or the current tool is completed
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_colon=1,
                    stop_after_opening_curly_braces=1,
                    # if the tool ends, we want to separate
                    # at the start of the next tool
                )
            elif self.streaming_state == StreamingState.PARSING_NAME:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_comma=1,
                    stop_after_closing_brackets=1,
                )
            elif self.streaming_state ==\
                StreamingState.WAITING_FOR_ARGUMENTS_START:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_opening_curly_braces=1,
                )
            elif self.streaming_state ==\
                StreamingState.PARSING_ARGUMENTS:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_closing_curly_braces=1,
                    # we could be more clever
                    # by listening to item.arguments.* start_map events
                    # and know how many curly braces we can allow
                )
            elif self.streaming_state in [
                    StreamingState.PARSING_ARGUMENTS_COMPLETED,
                    StreamingState.PARSING_NAME_COMPLETED
            ]:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_closing_curly_braces=1,
                    stop_after_closing_brackets=1,
                )
            elif self.streaming_state == StreamingState.TOOL_COMPLETE:
                delta_to_be_parsed, delta_text = self._split_delta(
                    delta_text=delta_text,
                    stop_after_opening_curly_braces=1,
                    stop_after_closing_brackets=1,
                )
            elif self.streaming_state == StreamingState.ALL_TOOLS_COMPLETE:
                content = delta_text
                delta_text = ""
            else:
                delta_to_be_parsed = delta_text
                delta_text = ""

            if self.streaming_state != StreamingState.ALL_TOOLS_COMPLETE:
                self.parse_coro.send(delta_to_be_parsed.encode('utf-8'))

            # Given the parsed text and the possible streaming state change,
            # let's add to the tool delta
            if ((streaming_state_before_parse != self.streaming_state)
                    and streaming_state_before_parse in [
                        StreamingState.WAITING_FOR_TOOL_START,
                        StreamingState.TOOL_COMPLETE
                    ] and self.streaming_state not in [
                        StreamingState.ALL_TOOLS_COMPLETE,
                        StreamingState.TOOL_COMPLETE,
                        StreamingState.WAITING_FOR_TOOL_START
                    ]):
                # starting a new tool call
                if current_tool_call_modified:
                    if self.current_tool_mistral_id is not None:
                        current_tool_call.id = self.current_tool_mistral_id
                        self.current_tool_mistral_id = None
                    delta_tool_calls.append(current_tool_call)
                current_tool_call_modified = False
                self.current_tool_id += 1
                self.current_tool_mistral_id = \
                    MistralToolCall.generate_random_id()
                current_tool_call = DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                )
            if current_tool_call.function is None:
                current_tool_call.function = DeltaFunctionCall()

            if self.current_tool_name is not None:
                # we have the complete tool name
                current_tool_call_modified = True
                current_tool_call.function.name = self.current_tool_name
                self.current_tool_name = None
            if self.streaming_state == StreamingState.PARSING_NAME_COMPLETED:
                self.streaming_state = StreamingState.WAITING_FOR_TOOL_KEY
            if self.streaming_state in [
                    StreamingState.PARSING_ARGUMENTS,
                    StreamingState.PARSING_ARGUMENTS_COMPLETED
            ]:
                if self.streaming_state ==\
                    StreamingState.PARSING_ARGUMENTS_COMPLETED:
                    self.streaming_state = StreamingState.WAITING_FOR_TOOL_KEY
                # the delta_to_be_parsed is part of arguments.
                current_tool_call_modified = True
                if current_tool_call.function.arguments is None:
                    current_tool_call.function.arguments = delta_to_be_parsed
                else:
                    current_tool_call.function.arguments += delta_to_be_parsed
                if streaming_state_before_parse !=\
                    StreamingState.PARSING_ARGUMENTS:
                    # It's the first chunk of arg. let's lstrip it
                    current_tool_call.function.arguments =\
                        current_tool_call.function.arguments.lstrip()

        if current_tool_call_modified:
            if self.current_tool_mistral_id is not None:
                current_tool_call.id = self.current_tool_mistral_id
                self.current_tool_mistral_id = None
            delta_tool_calls.append(current_tool_call)

        # HACK: serving_chat.py inspects the internal state of tool parsers
        # when determining it's final streaming delta, automatically
        # adding autocompleted JSON.
        # These two lines avoid that nonsense while ensuring finish_reason
        # is set to tool_calls when at least one tool is called.
        if delta_tool_calls and not self.prev_tool_call_arr:
            self.prev_tool_call_arr = [{"arguments": {}}]

        if content or len(delta_tool_calls) > 0:
            delta_message = DeltaMessage()
            if content:
                delta_message.content = content
            if len(delta_tool_calls) > 0:
                delta_message.tool_calls = delta_tool_calls
            return delta_message
        else:
            if self.streaming_state == StreamingState.ALL_TOOLS_COMPLETE:
                return DeltaMessage()
            else:
                return None

    def _split_delta(
        self,
        delta_text: str,
        stop_after_quotes: int = -1,
        stop_after_opening_curly_braces: int = -1,
        stop_after_closing_curly_braces: int = -1,
        stop_after_closing_brackets: int = -1,
        stop_after_colon: int = -1,
        stop_after_comma=-1,
    ) -> tuple[str, str]:
        delta_to_be_parsed = ""
        for i, c in enumerate(delta_text):
            if c in ['"', "'"]:
                delta_to_be_parsed += c
                stop_after_quotes -= 1
                if stop_after_quotes == 0:
                    return (delta_to_be_parsed, delta_text[i + 1:])
            elif c == "{":
                delta_to_be_parsed += c
                stop_after_opening_curly_braces -= 1
                if stop_after_opening_curly_braces == 0:
                    return (delta_to_be_parsed, delta_text[i + 1:])
            elif c == "}":
                delta_to_be_parsed += c
                stop_after_closing_curly_braces -= 1
                if stop_after_closing_curly_braces == 0:
                    return (delta_to_be_parsed, delta_text[i + 1:])
            elif c == "]":
                delta_to_be_parsed += c
                stop_after_closing_brackets -= 1
                if stop_after_closing_brackets == 0:
                    return (delta_to_be_parsed, delta_text[i + 1:])
            elif c == ":":
                delta_to_be_parsed += c
                stop_after_colon -= 1
                if stop_after_colon == 0:
                    return (delta_to_be_parsed, delta_text[i + 1:])
            elif c == ",":
                delta_to_be_parsed += c
                stop_after_comma -= 1
                if stop_after_comma == 0:
                    return (delta_to_be_parsed, delta_text[i + 1:])
            else:
                delta_to_be_parsed += c

        return (delta_to_be_parsed, "")
