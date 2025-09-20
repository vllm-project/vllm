# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from enum import Enum
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


class StreamingState(Enum):
    """Enum for tracking the current streaming parsing state."""
    WAITING_FOR_TOOL_START = "waiting_for_tool_start"
    PARSING_NAME = "parsing_name"
    PARSING_ARGUMENTS = "parsing_arguments"
    TOOL_COMPLETE = "tool_complete"
    ALL_TOOLS_COMPLETE = "all_tools_complete"


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

        # Optimized regex patterns
        self.tool_call_first_attribute_name: re.Pattern[str] = re.compile(
            r'.*\s*"name"\s*:\s*')
        self.string_value_pattern: re.Pattern[str] = re.compile(
            r'\s*"(.*?)(?<!\\)"')
        # - Lazy quantifier (.*?) to stop at first unescaped quote
        # - Negative lookbehind (?<!\\) to avoid escaped quotes
        self.tool_call_first_attribute_arguments: re.Pattern[str] = re.compile(
            r'.*\s*"arguments"\s*:\s*{')

        # Core streaming state
        self.raw_tool_calls: str = ""
        self.streaming_state: StreamingState = \
            StreamingState.WAITING_FOR_TOOL_START

        # Tool tracking
        self.current_tool_id: int = -1
        self.current_tool_start_index: int = -1
        self.current_attribute_start_index: int = -1
        self.previous_attribute_end_index: int = 0

        # Legacy state tracking (kept for compatibility)
        self.current_element_streaming: Union[Literal["name", "arguments"],
                                              None] = None
        self.current_tool_name_finished: bool = False
        self.current_tool_arguments_finished: bool = False
        self.tools_parsing_finished: bool = False

        # V11 format state
        self.v11_tool_format: bool = False
        self.current_tool_name_sent: bool = False
        self.prev_args_sent: str = ""

        # Caching for performance
        self._last_json_parse_input: str = ""
        self._last_json_parse_result: tuple[bool, int] = (False, -1)
        self.bot_token = "[TOOL_CALLS]"
        self.bot_token_id = self.vocab.get(self.bot_token)
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        if _is_fn_name_regex_support(self.model_tokenizer):
            self.fn_name_regex = re.compile(
                r'([a-zA-Z0-9_-]+)\s*(\{[\s\S]*?\})', re.DOTALL)
        else:
            self.fn_name_regex = None

        if self.bot_token_id is None:
            raise RuntimeError(
                "Mistral Tool Parser could not locate the tool call token in "
                "the tokenizer!")

    def _extract_tool_calls_streaming_v11(
            self, additional_content: str,
            delta_text: str) -> Union[DeltaMessage, None]:
        """
        Extract tool calls from a streaming response, specifically for the
        v11 MistralTokenizer format: ToolName{arguments}. This logic is a
        streaming equivalent of the `self.fn_name_regex` used in
        non-streaming extraction.
        """
        logger.debug("v11 streaming: raw_tool_calls='%s'", self.raw_tool_calls)
        logger.debug("v11 streaming: current_tool_name_sent='%s'",
                     self.current_tool_name_sent)
        logger.debug("v11 streaming: prev_args_sent='%s'", self.prev_args_sent)

        result_tool_calls: list[DeltaToolCall] = []

        while True:
            advanced = False
            if self.current_tool_name_finished and \
                self.current_tool_arguments_finished and \
                self._should_advance_to_next_v11_tool():
                # Remove the completed tool from raw_tool_calls
                # before resetting state
                completed_tool_end = self._find_completed_v11_tool_end()
                if completed_tool_end > 0:
                    self.raw_tool_calls = self.raw_tool_calls[
                        completed_tool_end:]
                self._reset_v11_tool_state()
                logger.debug("v11 streaming: found next tool, resetting state")
                advanced = True

            sent_something = False

            # Phase 1: Extract and send function name
            if not self.current_tool_name_sent:
                # Look for function name pattern: name followed by {
                brace_index = self.raw_tool_calls.find("{")
                if brace_index == -1:
                    logger.debug("v11 streaming: no opening brace found yet")
                    break

                # Extract function name
                func_name = self.raw_tool_calls[:brace_index].strip()
                # Remove any leading separators from previous tools
                func_name = re.sub(r'^[\s,]*', '', func_name)

                if not func_name:
                    logger.debug("v11 streaming: function name is empty")
                    break

                logger.debug("v11 streaming: sending function name='%s'",
                             func_name)
                self.current_tool_name_sent = True
                self.current_tool_name_finished = True
                self.current_tool_id += 1

                result_tool_calls.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=MistralToolCall.generate_random_id(),
                        function=DeltaFunctionCall(name=func_name).model_dump(
                            exclude_none=True),
                    ))
                sent_something = True

            # Phase 2: Extract and send argument fragments
            if self.current_tool_name_sent and \
                not self.current_tool_arguments_finished:
                # Find the arguments part (everything after the first {)
                brace_index = self.raw_tool_calls.find("{")
                if brace_index == -1:
                    logger.debug(
                        "v11 streaming: no opening brace found for args")
                    break

                current_args = self.raw_tool_calls[brace_index:]
                logger.debug("v11 streaming: current_args='%s'", current_args)

                actual_args = current_args
                try:
                    parsed_obj, end_idx = self.json_decoder.raw_decode(
                        current_args)
                    # JSON is complete
                    self.current_tool_arguments_finished = True
                    actual_args = current_args[:end_idx]
                    logger.debug("v11 streaming: JSON complete, parsed_obj=%s",
                                 parsed_obj)
                except json.decoder.JSONDecodeError:
                    # JSON still incomplete
                    logger.debug("v11 streaming: JSON still incomplete")
                    pass

                # Calculate what's new since last time
                new_content = ""
                if actual_args != self.prev_args_sent:
                    if self.prev_args_sent and actual_args.startswith(
                            self.prev_args_sent):
                        # Incremental update
                        new_content = actual_args[len(self.prev_args_sent):]
                        logger.debug("v11 streaming: incremental args='%s'",
                                     new_content)
                    else:
                        # First time or reset
                        new_content = actual_args
                        logger.debug("v11 streaming: first/reset args='%s'",
                                     new_content)

                self.prev_args_sent = actual_args

                if new_content:
                    result_tool_calls.append(
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=new_content).model_dump(
                                    exclude_none=True),
                        ))
                    sent_something = True

            if not sent_something and not advanced:
                break

        if result_tool_calls:
            return DeltaMessage(
                content=additional_content,
                tool_calls=result_tool_calls,
            )

        return self._none_or_additional_content(additional_content)

    def _should_advance_to_next_v11_tool(self) -> bool:
        """Check if we should advance to the next tool in V11 format."""
        completed_tool_end = self._find_completed_v11_tool_end()
        if completed_tool_end <= 0:
            return False

        # Check if there's content after the completed tool
        # that looks like another tool
        remaining = self.raw_tool_calls[completed_tool_end:].strip()
        if remaining.startswith(','):
            remaining = remaining[1:].strip()

        # Look for next tool pattern: function_name{
        return bool(re.match(r'[a-zA-Z0-9_-]+\s*\{', remaining))

    def _find_completed_v11_tool_end(self) -> int:
        """
        Find the end position of the first completed tool in V11 format
        using JSON parsing.
        """
        # Look for function name pattern: name followed by {
        brace_match = re.search(r'([a-zA-Z0-9_-]+)\s*(\{)',
                                self.raw_tool_calls)
        if not brace_match:
            return -1

        # Try to parse the JSON starting from the opening brace
        json_start = brace_match.start(2)
        json_part = self.raw_tool_calls[json_start:]

        try:
            _, end_idx = self.json_decoder.raw_decode(json_part)
            return json_start + end_idx
        except json.JSONDecodeError:
            return -1

    def _reset_v11_tool_state(self) -> None:
        """Reset V11 tool parsing state for the next tool."""
        self.current_tool_name_finished = False
        self.current_tool_arguments_finished = False
        self.current_tool_name_sent = False
        self.prev_args_sent = ""

    def _determine_next_parsing_element(self) \
        -> Union[Literal["name", "arguments"], None]:
        """
        Determine the next element to parse based on current state.
        
        Returns:
            The next element to parse, or None if nothing is ready
        """
        # Check for name attribute
        if not self.current_tool_name_finished:
            match_name = self.tool_call_first_attribute_name.match(
                self.raw_tool_calls, self.current_tool_start_index)
            if match_name and match_name.end(
            ) > self.current_tool_start_index \
                    + self.previous_attribute_end_index:
                self.current_attribute_start_index = match_name.end() \
                    - self.current_tool_start_index
                return "name"

        # Check for arguments attribute
        if not self.current_tool_arguments_finished:
            match_arguments = self.tool_call_first_attribute_arguments.match(
                self.raw_tool_calls, self.current_tool_start_index)
            if match_arguments and match_arguments.end(
            ) > self.current_tool_start_index \
                    + self.previous_attribute_end_index:
                # The `{` is the last character in the match.
                # We want it as start index.
                self.current_attribute_start_index = match_arguments.end() \
                    - 1 - self.current_tool_start_index
                return "arguments"

        return None

    def _is_current_tool_complete(self) -> bool:
        """Check if the current tool parsing is complete."""
        return (self.current_tool_name_finished
                and self.current_tool_arguments_finished)

    def _advance_to_next_tool(self) -> bool:
        """
        Advance to the next tool if available.
        
        Returns:
            True if successfully advanced to next tool, False otherwise
        """
        next_tool_start_index = self._next_tool_starting_position()
        if next_tool_start_index > 0:
            self.current_tool_id += 1
            self.current_tool_start_index = next_tool_start_index
            self.current_attribute_start_index = -1
            self.previous_attribute_end_index = 0
            self.current_tool_name_finished = False
            self.current_tool_arguments_finished = False
            return True
        return False

    def _process_delta_text(self, delta_text: str) -> str:
        """
        Process delta text and update raw_tool_calls, returning any additional
        content.
        
        Args:
            delta_text: The new text delta to process
            
        Returns:
            Any additional content that appears before the bot token
        """
        additional_content = ""

        if self.bot_token in delta_text:
            # Split only once for efficiency
            parts = delta_text.split(self.bot_token, 1)
            if len(parts) > 1:
                if parts[0]:  # Content before bot token
                    additional_content = parts[0]
                # Process content after bot token
                tool_content = parts[1].lstrip()
                self.raw_tool_calls += tool_content
        else:
            # No bot token in delta, just clean and append
            self.raw_tool_calls += delta_text
            # Remove leading spaces only if we have content
            if self.raw_tool_calls:
                self.raw_tool_calls = self.raw_tool_calls.lstrip()

        return additional_content

    def _should_detect_v11_format(self) -> bool:
        """Check if we should attempt V11 format detection."""
        return (self.fn_name_regex is not None and self.current_tool_id == -1
                and not self.v11_tool_format)

    def _detect_v11_format(self) -> None:
        """Detect if we're using V11 tool format."""
        stripped_calls = self.raw_tool_calls.lstrip()
        if stripped_calls and stripped_calls[0] != "[":
            logger.debug("flipping v11 tool format to True ...")
            self.v11_tool_format = True

    def _try_parse_json_cached(self, text: str) -> tuple[bool, int]:
        """
        Attempt to parse JSON with caching for performance.
        
        Args:
            text: The text to parse as JSON
            
        Returns:
            Tuple of (success, end_index)
        """
        if text == self._last_json_parse_input:
            return self._last_json_parse_result

        try:
            _, end_index = self.json_decoder.raw_decode(text)
            result = (True, end_index)
        except json.decoder.JSONDecodeError:
            result = (False, -1)

        # Cache the result
        self._last_json_parse_input = text
        self._last_json_parse_result = result
        return result

    def _extracted_complete_name(
            self, current_attribute_start_index: int) \
        -> tuple[str, Union[int, None]]:
        """
        Extract the complete function name from the current tool call.

        Args:
            current_attribute_start_index: The starting index of the
            name attribute relative to the current tool start

        Returns:
            tuple:
            - The function name, or "" if extraction failed
            - The end index of the name relative to the current tool start,
            or None if extraction failed
        """
        absolute_start = self.current_tool_start_index \
            + current_attribute_start_index
        if match := self.string_value_pattern.match(\
            self.raw_tool_calls, absolute_start):
            return match.group(1), match.end() - self.current_tool_start_index
        return "", None

    def _extract_argument_fragment(self, current_attribute_start_index: int,
                                   delta: str) -> tuple[str, int]:
        """
        Extract the relevant argument fragment from the current streaming delta.

        Args:
            current_attribute_start_index: The starting index
            of the arguments attribute relative to the current tool start
            delta: The new text added in this streaming step

        Returns:
            tuple:
            - The extracted argument diff text
            to be sent in the streaming response
            - The end index of the arguments relative to the current tool start,
            or -1 if not yet complete
        """
        absolute_start = self.current_tool_start_index \
            + current_attribute_start_index
        partial_arguments_value = self.raw_tool_calls[absolute_start:]
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
        try:
            _, end_index = self.json_decoder.raw_decode(
                self.raw_tool_calls, self.current_tool_start_index)
            # Look for the next opening brace after the current tool ends
            search_start = self.current_tool_start_index + end_index
            next_brace = self.raw_tool_calls.find("{", search_start)
            return next_brace if next_brace != -1 else -1
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
                    pos = 0
                    tool_str = tool_content
                    while pos < len(tool_str):
                        # skip ws
                        while pos < len(tool_str) and tool_str[pos].isspace():
                            pos += 1
                        if pos >= len(tool_str):
                            break

                        # match name
                        match_name = re.match(r'([a-zA-Z0-9_-]+)',
                                              tool_str[pos:])
                        if not match_name:
                            break
                        fn_name = match_name.group(0)
                        pos += match_name.end()

                        # skip ws
                        while pos < len(tool_str) and tool_str[pos].isspace():
                            pos += 1

                        if pos >= len(tool_str) or tool_str[pos] != '{':
                            break

                        pos += 1  # skip {

                        # parse args
                        try:
                            args_obj, end_idx = self.json_decoder.raw_decode(
                                tool_str[pos:])
                            function_call_arr.append({
                                "name": fn_name,
                                "arguments": args_obj
                            })
                            pos += end_idx
                        except json.JSONDecodeError:
                            break

                        # skip ws
                        while pos < len(tool_str) and tool_str[pos].isspace():
                            pos += 1

                        # optional comma
                        if pos < len(tool_str) and tool_str[pos] == ',':
                            pos += 1
                            while pos < len(
                                    tool_str) and tool_str[pos].isspace():
                                pos += 1
                else:
                    function_call_arr = json.loads(tool_content)
            except json.JSONDecodeError:
                # use a regex to find the part corresponding to the tool call.
                # NOTE: This use case should not happen if the model is trained
                # correctly. It's an easy possible fix so it's included, but
                # can be brittle for very complex / highly nested tool calls
                raw_tool_call = self.tool_call_regex.search(
                    tool_content).group(0)
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

        # Early return if no tool call token present
        if self.bot_token not in current_text:
            return DeltaMessage(content=delta_text)

        # Process delta text and extract additional content
        additional_content = self._process_delta_text(delta_text)

        # Detect and handle V11 format
        if self._should_detect_v11_format():
            self._detect_v11_format()

        if self.v11_tool_format:
            return self._extract_tool_calls_streaming_v11(
                additional_content, delta_text)

        # Check if tool calls have started
        if self.current_tool_start_index < 0:
            bracket_pos = self.raw_tool_calls.find("[")
            if bracket_pos >= 0:
                self.current_tool_start_index = bracket_pos + 1
                self.current_tool_id += 1
            else:
                return self._none_or_additional_content(additional_content)

        # Try to parse complete JSON with caching
        parse_success, end_index = self._try_parse_json_cached(
            self.raw_tool_calls)
        if parse_success:
            self.tools_parsing_finished = True
            if len(self.raw_tool_calls) > end_index:
                additional_content = self.raw_tool_calls[end_index:]

        # Handle tool completion and transition to next tool
        if self._is_current_tool_complete():
            if self.tools_parsing_finished:
                return self._none_or_additional_content(additional_content)

            if self._advance_to_next_tool():
                # Successfully moved to next tool, continue processing
                pass
            else:
                # No next tool ready yet
                return self._none_or_additional_content(additional_content)

        if self.current_tool_start_index >= len(self.raw_tool_calls):
            # tool call has not started
            return self._none_or_additional_content(additional_content)

        # Determine what to parse next
        if self.current_element_streaming is None:
            next_element = self._determine_next_parsing_element()
            if next_element is None:
                return self._none_or_additional_content(additional_content)
            self.current_element_streaming = next_element

        if self.current_element_streaming == "name":
            try:
                function_name, name_end_index = self._extracted_complete_name(
                    self.current_attribute_start_index)
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
