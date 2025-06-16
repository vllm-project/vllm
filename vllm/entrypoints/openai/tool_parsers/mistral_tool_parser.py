# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits
from typing import Literal, Union

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


def _is_fn_name_regex_support(model_tokenizer: AnyTokenizer) -> bool:
    return isinstance(model_tokenizer, MistralTokenizer) \
        and model_tokenizer.version >= 11


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
        self.json_decoder: json.JSONDecoder = json.JSONDecoder()
        self.tool_call_first_attribute_name: re.Pattern[str] = re.compile(
            r'.*\s*"name"\s*:\s*')
        self.string_value_pattern: re.Pattern[str] = re.compile(
            r'\s*"(.*?)(?<!\\)"')
        # - Lazy quantifier (.*?) to stop at first unescaped quote
        # - Negative lookbehind (?<!\\) to avoid escaped quotes
        self.tool_call_first_attribute_arguments: re.Pattern[str] = re.compile(
            r'.*\s*"arguments"\s*:\s*{')

        self.raw_tool_calls: str = ""
        self.tools_parsing_finished: bool = False

        self.current_tool_id: int = -1
        self.current_tool_start_index: int = -1
        # index in the `self.raw_tool_calls` string
        self.current_attribute_start_index: int = -1
        # index in the `self.raw_current_tool_call` string
        self.previous_attribute_end_index: int = 0
        # index in the `self.raw_current_tool_call` string
        self.current_element_streaming: Union[Literal["name", "arguments"],
                                              None] = None
        self.current_tool_name_finished: bool = False
        self.current_tool_arguments_finished: bool = False

        self.bot_token = "[TOOL_CALLS]"
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        if _is_fn_name_regex_support(self.model_tokenizer):
            self.fn_name_regex = re.compile(r'([a-zA-Z0-9_-]+)(\{.*?\})',
                                            re.DOTALL)
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
                    matches = self.fn_name_regex.findall(tool_content)

                    function_call_arr = []
                    for match in matches:
                        fn_name = match[0]
                        args = match[1]

                        # fn_name is encoded outside serialized json dump
                        # only arguments are serialized
                        function_call_arr.append({
                            "name": fn_name,
                            "arguments": json.loads(args)
                        })
                else:
                    function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                # use a regex to find the part corresponding to the tool call.
                # NOTE: This use case should not happen if the model is trained
                # correctly. It's a easy possible fix so it's included, but
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

        # if the tool call token is not in the tokens generated so far, append
        # output to contents since it's not a tool
        if self.bot_token not in current_text:
            return DeltaMessage(content=delta_text)

        # if the tool call token ID IS in the tokens generated so far, that
        # means we're parsing as tool calls now
        additional_content: str = ""
        if self.bot_token in delta_text:
            self.raw_tool_calls += (delta_text.split(
                self.bot_token)[-1].replace("'", '"').lstrip())
            if not delta_text.startswith(self.bot_token):
                # delta contains some text before the bot token
                additional_content = delta_text.split(self.bot_token)[0]
        else:
            self.raw_tool_calls += delta_text.replace("\'", "\"")
            self.raw_tool_calls = (
                self.raw_tool_calls.lstrip()
            )  # leading spaces prevent us from raw_decoding

        if self.current_tool_start_index < 0:
            if "[" in self.raw_tool_calls:
                self.current_tool_start_index = self.raw_tool_calls.find(
                    "[") + 1
                self.current_tool_id += 1
            else:
                # tool calls not started
                return self._none_or_additional_content(additional_content)

        try:
            _, end_index = self.json_decoder.raw_decode(self.raw_tool_calls)
            self.tools_parsing_finished = True
            if len(self.raw_tool_calls) > end_index:
                additional_content = self.raw_tool_calls[end_index:]
        except json.decoder.JSONDecodeError:
            # we are in tool calls
            pass

        if (self.current_tool_name_finished
                and self.current_tool_arguments_finished):
            if self.tools_parsing_finished:
                return self._none_or_additional_content(additional_content)
            # let's find the next tool starting position
            next_tool_start_index = self._next_tool_starting_position()
            if next_tool_start_index > 0:
                self.current_tool_id += 1
                self.current_tool_start_index = next_tool_start_index
                self.current_attribute_start_index = -1
                self.previous_attribute_end_index = 0
                self.current_tool_name_finished = False
                self.current_tool_arguments_finished = False

        if self.current_tool_start_index >= len(self.raw_tool_calls):
            # tool call has not started
            return self._none_or_additional_content(additional_content)
        raw_current_tool_call = self.raw_tool_calls[self.
                                                    current_tool_start_index:]

        if self.current_element_streaming is None:
            # we are waiting for the next argument to be parsed
            match_name = self.tool_call_first_attribute_name.match(
                raw_current_tool_call)
            match_arguments = self.tool_call_first_attribute_arguments.match(
                raw_current_tool_call)
            if not self.current_tool_name_finished and match_name:
                if match_name.end() <= self.previous_attribute_end_index:
                    return self._none_or_additional_content(additional_content)
                self.current_element_streaming = "name"
                self.current_attribute_start_index = match_name.end()
            elif not self.current_tool_arguments_finished and match_arguments:
                if match_arguments.end() <= self.previous_attribute_end_index:
                    return self._none_or_additional_content(additional_content)
                self.current_element_streaming = "arguments"
                self.current_attribute_start_index = match_arguments.end() - 1
                # the `{` is the last IN the match part.
                # We want it as the start index element
            else:
                # let's wait for more deltas
                return self._none_or_additional_content(additional_content)

        if self.current_element_streaming == "name":
            try:
                function_name, name_end_index = self._extracted_complete_name(
                    raw_current_tool_call, self.current_attribute_start_index)
            except IndexError:
                # name value has not started being generated
                return self._none_or_additional_content(additional_content)
            if function_name == "":
                return self._none_or_additional_content(additional_content)
            else:
                assert name_end_index is not None
                # because the function name was successfully retrieved

                self.current_tool_name_finished = True
                self.current_element_streaming = None
                self.current_attribute_start_index = -1
                self.previous_attribute_end_index = name_end_index
                delta = DeltaMessage(
                    content=additional_content,
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=MistralToolCall.generate_random_id(),
                            function=DeltaFunctionCall(
                                name=function_name).model_dump(
                                    exclude_none=True),
                        )
                    ],
                )
                return delta
        if self.current_element_streaming == "arguments":
            try:
                diff, arguments_end_index = self._extract_argument_fragment(
                    raw_current_tool_call,
                    self.current_attribute_start_index,
                    delta_text,
                )
                self.current_tool_arguments_finished = arguments_end_index != -1
                if self.current_tool_arguments_finished:
                    self.current_element_streaming = None
                    self.current_attribute_start_index = -1
                    self.previous_attribute_end_index = arguments_end_index
                delta = DeltaMessage(
                    content=additional_content,
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=diff).model_dump(exclude_none=True),
                        )
                    ],
                )
                return delta
            except IndexError:
                # arguments value has not started being generated
                return self._none_or_additional_content(additional_content)

    def _extracted_complete_name(
            self, raw_current_tool_call: str,
            current_attribute_start_index: int
    ) -> tuple[str, Union[int, None]]:
        """
        Extract the complete function name from the current tool call.

        Args:
            raw_current_tool_call: The raw JSON string of the current tool call
            current_attribute_start_index: The starting index of the
            name attribute in the raw_current_tool_call string

        Returns:
            tuple:
            - The function name, or "" if extraction failed
            - The end index of the name in raw_current_tool_call,
            or None if extraction failed
        """
        partial_name_value = raw_current_tool_call[
            current_attribute_start_index:]
        if match := self.string_value_pattern.match(partial_name_value):
            return match.group(1), match.end() + current_attribute_start_index
        return "", None

    def _extract_argument_fragment(self, raw_current_tool_call: str,
                                   current_attribute_start_index: int,
                                   delta: str) -> tuple[str, int]:
        """
        Extract the relevant argument fragment from the current streaming delta.

        Args:
            raw_current_tool_call: The raw JSON string of the current tool call
            current_attribute_start_index: The starting index
            of the arguments attribute in the raw string
            delta: The new text added in this streaming step

        Returns:
            tuple:
            - The extracted argument diff text
            to be sent in the streaming response
            - The end index of the arguments in the raw string,
            or -1 if not yet complete
        """
        partial_arguments_value = raw_current_tool_call[
            current_attribute_start_index:]
        try:
            _, end_index = self.json_decoder.raw_decode(
                partial_arguments_value)
            return (
                delta[:len(delta) + end_index - len(partial_arguments_value)],
                current_attribute_start_index + end_index,
            )
        except json.decoder.JSONDecodeError:
            # The arguments object is not complete

            # delta contains data from before the argument start
            if len(delta) > len(partial_arguments_value):
                return delta[-len(partial_arguments_value):], -1

            # We can send the whole delta
            return delta, -1

    def _next_tool_starting_position(self) -> int:
        """
        Find the starting position of the next tool
        in the raw tool calls string.

        Returns:
            The index position where the next tool starts,
            or -1 if no next tool is found yet
        """
        assert self.current_tool_start_index >= 0
        current_tool_call = self.raw_tool_calls[self.current_tool_start_index:]
        try:
            _, end_index = self.json_decoder.raw_decode(current_tool_call)
            return (self.current_tool_start_index + end_index +
                    current_tool_call[end_index:].find("{"))
        except json.decoder.JSONDecodeError:
            # The current tool object is not yet closed
            return -1
        except IndexError:
            # The next tool has not started yet
            # and the delta just closes the current tool call
            return -1

    def _none_or_additional_content(
            self, additional_content: str) -> Union[DeltaMessage, None]:
        """
        Create a DeltaMessage with additional content if present,
        otherwise return None.

        Args:
            additional_content: The text content to include in the message

        Returns:
            A DeltaMessage with the additional content,
            or None if no content is provided
        """
        if additional_content:
            return DeltaMessage(content=additional_content)
        return None
