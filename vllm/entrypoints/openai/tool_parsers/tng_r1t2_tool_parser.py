# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
import json
from collections.abc import Sequence
from enum import Enum
from typing import Any, Union

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


class ParsedStructure(Enum):
    CONTENT = 1
    REASONING_CONTENT = 2
    TOOL_CALL = 3
    TOOL_CALL_DELIMITER = 4
    TOOL_CALL_START_TAG = 5
    TOOL_CALL_END_TAG = 6


@ToolParserManager.register_module("tng_r1t2")
class TngR1T2ToolParser(ToolParser):
    """Tool Parser for models like tngtech/DeepSeek-TNG-R1T2-Chimera,
    It is compatible with hermes tool call templates
    but does not require <tool_call> and </tool_call>
    to be single tokens in the vocabulary;
    instead only the string representation of the model output
    is parsed, making this tool call parser robust and versatile."""

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # For backward compatibility with serving code
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        self.think_tag_pattern = r"(<think>[\s\S]*?</think>)"
        self.think_start_tag = "<think>"
        self.think_end_tag = "</think>"

        self.tool_call_tag_pattern = r"<tool_call>([\s\S]*?)</tool_call>"
        self.tool_call_start_tag = "<tool_call>"
        self.tool_call_end_tag = "</tool_call>"

        # Define streaming state type to be initialized later
        self.streaming_state: dict[str, Any] = {
            "streamed_tool_calls": [],
            "buffer": "",
            "parsed_structure": ParsedStructure.CONTENT,
        }

    def extract_tool_call_from_nonthink_output(
            self, raw_text: str) -> tuple[str, list[dict]]:
        parts = re.split(self.tool_call_tag_pattern, raw_text)
        content = ""
        tool_calls: list[dict] = []
        for i, part in enumerate(parts):
            is_potential_tool_call = i % 2 == 1
            if is_potential_tool_call:
                try:
                    more_tool_calls = json.loads(part)
                    if isinstance(more_tool_calls, list):
                        tool_calls.extend(more_tool_calls)
                    else:
                        tool_calls.extend([more_tool_calls])
                except json.JSONDecodeError:
                    logger.warning("Invalid tool call json "
                                   "-> parse as text content")
                    content += part
                    continue
            else:
                content += part
        return content, tool_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model output.
        """
        # split at think traces -> those will not be parsed for tool calls
        think_parts = re.split(self.think_tag_pattern, model_output)
        content = ""
        tool_calls = []
        for i, part in enumerate(think_parts):
            parse_output = i % 2 == 0
            if parse_output:
                more_content, more_tool_calls = (
                    self.extract_tool_call_from_nonthink_output(part))
                content += more_content
                tool_calls += more_tool_calls
            else:
                content += part

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=content,
            )

        tool_call_objs: list[ToolCall] = []

        for idx, call in enumerate(tool_calls):
            if (not isinstance(call, dict) or "name" not in call
                    or "arguments" not in call):
                logger.warning("Invalid tool call format, ignore.")
                continue

            tool_call = ToolCall(
                id=f"call_{idx}_{random_uuid()}",
                type="function",
                function=FunctionCall(
                    name=call["name"],
                    arguments=(json.dumps(call["arguments"]) if isinstance(
                        call["arguments"], dict) else call["arguments"]),
                ),
            )
            tool_call_objs.append(tool_call)

        return ExtractedToolCallInformation(
            tools_called=len(tool_call_objs) > 0,
            tool_calls=tool_call_objs,
            content=content,
        )

    def _parse_think_trace(self, raw_text: str) -> tuple[str, bool, str]:
        """
        Returns: (unambiguous_text_content, found_think_end, rest_string)
        """
        # Either a complete think_end_tag can be somewhere in raw_text
        think_end_pos = raw_text.find(self.think_end_tag)
        if think_end_pos >= 0:
            # in contrast to tool_call_start_tags, </think> remains part of content
            think_end_tag_end_pos = think_end_pos + len(self.think_end_tag)
            return (raw_text[:think_end_tag_end_pos], True,
                    raw_text[think_end_tag_end_pos:])

        # or the end of raw_text can be continued to a complete think_end_tag
        think_end_pos = (
            len(raw_text) -
            self._ends_with_partial_token(raw_text, self.think_end_tag))
        return raw_text[:think_end_pos], False, raw_text[think_end_pos:]

    def _parse_unambiguous_text_content(
            self, raw_text: str) -> tuple[str, Union[str, None], str]:
        """
        Returns: (unambiguous_text_content, interrupting_tag, rest_string)
        """
        # Either a complete tool_call_start_tag or think_start can be somewhere in raw_text
        search_tags = [self.think_start_tag, self.tool_call_start_tag]
        tag_positions = [(tag, pos) for tag in search_tags
                         if (pos := raw_text.find(tag)) >= 0]
        tag_positions.sort(key=lambda tag_and_pos: tag_and_pos[1])
        if len(tag_positions) > 0:
            first_tag, tag_pos = tag_positions[0]
            return raw_text[:tag_pos], first_tag, raw_text[tag_pos:]

        # or the end of raw_text can be continued to a complete tag
        tag_positions = [
            (tag, len(raw_text) - self._ends_with_partial_token(raw_text, tag))
            for tag in search_tags
        ]
        tag_positions.sort(key=lambda tag_and_pos: tag_and_pos[1])
        first_tag, tag_pos = tag_positions[0]
        if tag_pos < len(raw_text):
            return raw_text[:tag_pos], None, raw_text[tag_pos:]
        return raw_text, None, ""

    def _parse_tool_call_start_tag(self, raw_text: str) -> tuple[bool, str]:
        """
        Removes tool_call_start_tag from the beginning of raw_text,
        and an optional "[", and leading whitespace.
        Returns: (found_complete_tool_call_start_tag, rest_string)
        """
        if not raw_text.startswith(self.tool_call_start_tag):
            return False, raw_text
        rest = raw_text[len(self.tool_call_start_tag):].lstrip()
        if rest.startswith("["):
            rest = rest[1:].lstrip()
        return True, rest

    def _parse_tool_call_end_tag(
            self, raw_text: str) -> tuple[Union[bool, None], str]:
        """
        Removes tool_call_end_tag from the beginning of raw_text,
        and an optional "]" before it, and leading whitespace.
        Returns: tuple
            found_complete_tool_call_end_tag (or None if not decidable yet)
            rest_string
        """
        # remove optional whitespace and closing ] bracket from json list notation
        rest = raw_text.lstrip()
        if rest.startswith("]"):
            rest = rest[1:].lstrip()

        if rest.startswith(self.tool_call_end_tag):
            # found a complete tool call end tag
            return True, rest[len(self.tool_call_end_tag):]
        if (len(rest) >= len(self.tool_call_end_tag)
                or rest != self.tool_call_end_tag[:len(rest)]):
            # evidence that rest_string does not start with a tool call end tag
            return False, raw_text
        # incomplete tool call end tag, can not be decided yet
        return None, raw_text

    def _extract_arguments_from_partial_tool_call(
            self, raw_text: str) -> Union[str, None]:
        """
        Extracts the raw text of the "arguments" field of a complete
        or partial tool call.
        Args:
            raw_text: tool call raw text,
                      e.g `{"name": "my_tool", "arguments": {"firstarg": "some`

        Returns:
            raw text of the "arguments" field, which is not valid JSON
            unless the tool call is complete,
            e.g. `{"firstarg": "some` for the example raw_text above
        """
        # assumptions:
        # - "arguments" is always an object
        # - there is no other field of type object in the function call
        # - `raw_text` contains first "name", then "arguments" (otherwise,
        #   we'd have to find the end of "arguments" before returning its
        #   raw text value)

        # typically, at position 0, but there might be leading whitespace
        tool_call_start_pos = raw_text.find("{")
        assert raw_text[:tool_call_start_pos].strip() == ""
        arguments_start_pos = raw_text.find("{", tool_call_start_pos + 1)
        if arguments_start_pos < 0:
            return None
        arguments_raw_text = raw_text[arguments_start_pos:]
        return arguments_raw_text

    def _parse_complete_tool_call(
            self, raw_text: str) -> tuple[Union[dict, None], str]:
        """
        Returns: tuple
            parsed tool call if complete, None otherwise
            rest_string that needs to be parsed again or may contain
                        a partial tool call
        """
        # raw_text must start without whitespace for correct parsing
        obj, end_pos = self.extract_complete_json_dict(raw_text)
        if obj is None:
            return None, raw_text
        tool_call_raw_text = raw_text[:end_pos]
        # `tool_call_raw_text` is something like:
        #   '{"name": "tool-name", "arguments": {...xyz...} }'
        # we want to extract `{...xyz...}`,
        # but `extract_arguments_from_partial_tool_call` would return
        # everything after the second '{', i.e. `{...xyz...} }`
        arguments_raw_text = self._extract_arguments_from_partial_tool_call(
            tool_call_raw_text.removesuffix("}").rstrip())
        tool_call = {
            "name": obj.get("name"),
            "arguments": obj.get("arguments"),
            "arguments_raw_text": arguments_raw_text,
            "is_complete": True,
        }
        return tool_call, raw_text[end_pos:]

    def _parse_partial_tool_call(self, raw_text: str) -> Union[dict, None]:
        # raw_text must start without whitespace for correct parsing
        obj = partial_json_parser.loads(raw_text, Allow.ALL)
        arguments_raw_text = (
            self._extract_arguments_from_partial_tool_call(raw_text))
        tool_call = {
            "name": obj.get("name"),
            "arguments": obj.get("arguments"),
            "arguments_raw_text": arguments_raw_text,
            "is_complete": False,
        }
        return tool_call

    def _parse_tool_call(
            self,
            raw_text: str) -> tuple[Union[bool, None], Union[dict, None], str]:
        # remove optional whitespace and closing ] bracket
        # from json list notation
        rest = raw_text.lstrip()
        if rest == "":
            # no json has been received yet
            # -> can't tell if this will be a valid tool call
            return None, None, raw_text
        if not rest.startswith("{"):
            # can't be a tool call json
            return False, None, raw_text

        tool_call, rest = self._parse_complete_tool_call(rest)
        if tool_call:
            return True, tool_call, rest

        try:
            tool_call = self._parse_partial_tool_call(rest)
            # need to re-parse partial tool call later again
            # -> return None, not True
            return None, tool_call, rest
        except json.JSONDecodeError:
            # invalid json -> neither complete nor partial tool call
            return False, None, rest

    def _parse_tool_call_delimiter(
            self, raw_text: str) -> tuple[Union[bool, None], str]:
        """
        Returns: tuple
            does raw_text start with tool call delimiter?
                (None if undecidable/incomplete)
            rest_string
        """
        rest = raw_text.lstrip()
        if rest == "":
            return None, raw_text
        has_next_tool_call = rest.startswith(",")
        if not has_next_tool_call:
            return False, raw_text

        rest = rest[1:].lstrip()
        if rest == "":
            return None, raw_text
        has_next_tool_call = rest.startswith("{")
        if not has_next_tool_call:
            return False, raw_text
        return True, rest

    def _parse_all(
        self, raw_text: str, start_mode: ParsedStructure
    ) -> tuple[str, list[dict], str, ParsedStructure]:
        if start_mode == ParsedStructure.REASONING_CONTENT:
            content, found_closing_think, rest = self._parse_think_trace(
                raw_text)
            if found_closing_think:
                more_content, tool_calls, rest, structure = self._parse_all(
                    rest, start_mode=ParsedStructure.CONTENT)
                return content + more_content, tool_calls, rest, structure
            return content, [], rest, ParsedStructure.REASONING_CONTENT

        elif start_mode == ParsedStructure.CONTENT:
            content, interrupting_tag, rest = self._parse_unambiguous_text_content(
                raw_text)

            # rest might contain a tool call start tag or a think start tag
            if interrupting_tag == self.tool_call_start_tag:
                more_content, tool_calls, rest, structure = self._parse_all(
                    rest, start_mode=ParsedStructure.TOOL_CALL_START_TAG)
                return content + more_content, tool_calls, rest, structure
            elif interrupting_tag == self.think_start_tag:
                more_content, tool_calls, rest, structure = self._parse_all(
                    rest, start_mode=ParsedStructure.REASONING_CONTENT)
                return content + more_content, tool_calls, rest, structure
            else:
                return content, [], rest, ParsedStructure.CONTENT

        elif start_mode == ParsedStructure.TOOL_CALL_START_TAG:
            found_tool_call_start_tag, rest = self._parse_tool_call_start_tag(
                raw_text)
            if not found_tool_call_start_tag:
                return "", [], raw_text, ParsedStructure.CONTENT
            # we found a complete start tag, but we haven't seen the begin of a tool call json yet
            content, tool_calls, rest, structure = self._parse_all(
                rest, start_mode=ParsedStructure.TOOL_CALL)
            if not content and not tool_calls:
                # We haven't reached the opening "{" of the tool call yet.
                # We might see a "[" before the "{", so let's process the start tag again next chunk.
                return content, [], raw_text, ParsedStructure.CONTENT
            return content, tool_calls, rest, structure

        elif start_mode == ParsedStructure.TOOL_CALL:
            found_tool_call, tool_call, rest = self._parse_tool_call(raw_text)
            if found_tool_call is True:
                tool_calls = [tool_call] if tool_call else []
                content, more_tool_calls, rest, structure = self._parse_all(
                    rest, start_mode=ParsedStructure.TOOL_CALL_DELIMITER)
                return (content, tool_calls + more_tool_calls, rest, structure)
            elif found_tool_call is None:
                # partial tool call -> need to parse again with next chunk
                tool_calls = ([tool_call] if tool_call is not None else [])
                return "", tool_calls, rest, ParsedStructure.TOOL_CALL
            else:
                logger.warning(
                    "Invalid tool call -> continue with parsing model output as text content"
                )
                return self._parse_all(raw_text,
                                       start_mode=ParsedStructure.CONTENT)

        elif start_mode == ParsedStructure.TOOL_CALL_DELIMITER:
            found_tool_call_delimiter, rest = self._parse_tool_call_delimiter(
                raw_text)
            if found_tool_call_delimiter is True:
                return self._parse_all(rest,
                                       start_mode=ParsedStructure.TOOL_CALL)
            elif found_tool_call_delimiter is None:
                # could neither confirm nor deny that raw_text starts with a tool call delimiter
                return "", [], rest, ParsedStructure.TOOL_CALL_DELIMITER
            else:
                return self._parse_all(
                    raw_text, start_mode=ParsedStructure.TOOL_CALL_END_TAG)

        elif start_mode == ParsedStructure.TOOL_CALL_END_TAG:
            found_tool_call_end_tag, rest = self._parse_tool_call_end_tag(
                raw_text)
            if found_tool_call_end_tag is True:
                return self._parse_all(rest,
                                       start_mode=ParsedStructure.CONTENT)
            elif found_tool_call_end_tag is None:
                return "", [], rest, ParsedStructure.TOOL_CALL_END_TAG
            else:
                return self._parse_all(raw_text,
                                       start_mode=ParsedStructure.CONTENT)

        logger.warning(
            f"Unknown tool call parser start_mode '{start_mode}'. Falling back to text content."
        )
        return self._parse_all(raw_text, start_mode=ParsedStructure.CONTENT)

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
        """
        Extract tool calls for streaming mode.
        """
        raw_text = self.streaming_state["buffer"] + delta_text
        structure = self.streaming_state["parsed_structure"]

        content, tool_calls, rest, new_structure = self._parse_all(
            raw_text, start_mode=structure)
        self.streaming_state["buffer"] = rest
        self.streaming_state["parsed_structure"] = new_structure

        already_streamed_tool_calls = (
            self.streaming_state["streamed_tool_calls"])
        already_streamed_complete_tool_calls = [
            tool_call for tool_call in already_streamed_tool_calls
            if tool_call["is_complete"]
        ]
        all_tool_calls = (already_streamed_complete_tool_calls +
                          (tool_calls or []))
        to_be_streamed_tool_calls = self._calculate_delta_tool_calls(
            all_tool_calls, already_streamed_tool_calls)

        if not content and not to_be_streamed_tool_calls:
            return None
        self.update_state_vars(all_tool_calls)
        return DeltaMessage(content=content if content else None,
                            tool_calls=to_be_streamed_tool_calls)

    def _calculate_delta_tool_calls(
            self, current_tool_calls: Union[list[dict], None],
            already_streamed_tool_calls: list[dict]) -> list[DeltaToolCall]:
        if not current_tool_calls:
            return []

        new_deltas = []
        for tool_call_idx, partial_tool_call in enumerate(current_tool_calls):
            if (partial_tool_call.get("name") is None
                    or partial_tool_call.get("arguments") is None):
                # do not stream arguments for an unknown tool name;
                # and unless arguments appear in the partial json,
                # it might be that "name" has not been received completely
                # (assuming a template like `{"name": "mytool", "arguments": ...}`)
                continue
            partial_tool_call["tool_call_idx"] = tool_call_idx
            partial_tool_call["arguments_raw_text"] = (
                partial_tool_call.get('arguments_raw_text') or "")

            if len(already_streamed_tool_calls) > tool_call_idx:
                # parts of this tool_call_idx have already been streamed
                already_streamed_tool_call = already_streamed_tool_calls[
                    tool_call_idx]
                delta_tool_call = self._delta_for_partial_tool_call(
                    partial_tool_call, already_streamed_tool_call)
                if delta_tool_call is not None:
                    new_deltas.append(delta_tool_call)
                already_streamed_tool_calls[tool_call_idx] = (
                    already_streamed_tool_call | partial_tool_call)
            else:
                # no parts of this tool_call_idx have been streamed yet
                tool_call = self._delta_for_new_tool_call(partial_tool_call)
                new_deltas.append(tool_call)
                already_streamed_tool_calls.append(partial_tool_call)

        return new_deltas

    def _delta_for_new_tool_call(self, tool_call_dict: dict) -> DeltaToolCall:
        """constructs DeltaToolCall for new tool call,
        with tool_call_id, name, and all arguments seen so far.
        Updates tool_call dictionary with tool_call_id.
        """
        tool_call_idx = tool_call_dict["tool_call_idx"]
        tool_call_dict[
            "tool_call_id"] = f"call_{tool_call_idx}_{random_uuid()}"
        tool_call_dict["arguments_raw_text"] = tool_call_dict.get(
            'arguments_raw_text') or ""
        delta_tool_call = DeltaToolCall(
            index=tool_call_idx,
            type="function",
            id=tool_call_dict["tool_call_id"],
            function=DeltaFunctionCall(
                name=tool_call_dict.get("name"),
                arguments=tool_call_dict["arguments_raw_text"]))
        return delta_tool_call

    def _delta_for_partial_tool_call(
            self, new_tool_call: dict,
            already_streamed_tool_call: dict) -> Union[DeltaToolCall, None]:
        """Calculate delta for a tool call of which some parts have already been streamed."""
        assert new_tool_call["name"] == already_streamed_tool_call["name"]
        assert already_streamed_tool_call.get("tool_call_id")
        if already_streamed_tool_call.get("is_complete"):
            return None

        to_be_streamed_arguments = (
            new_tool_call["arguments_raw_text"].removeprefix(
                already_streamed_tool_call["arguments_raw_text"]))
        if not to_be_streamed_arguments:
            return None

        delta_tool_call = DeltaToolCall(
            index=new_tool_call["tool_call_idx"],
            type="function",
            function=DeltaFunctionCall(arguments=to_be_streamed_arguments))
        return delta_tool_call

    def update_state_vars(self, all_tools: list[dict]) -> None:
        # `tool_parser.streamed_args_for_tool` and
        # `tool_parser.prev_tool_call_arr` are checked in serving_chat.py

        # relevant is {"arguments": {...}}
        self.prev_tool_call_arr = all_tools

        # json-serialized argument
        self.streamed_args_for_tool = [
            tool_call.get("arguments_raw_text", "") for tool_call in all_tools
        ]

    @classmethod
    def _ends_with_partial_token(cls, buffer: str, tag: str) -> int:
        """
        Check if buffer ends with a partial tag.
        Return the length of the partial tag.
        """
        for i in range(1, min(len(buffer) + 1, len(tag))):
            if tag.startswith(buffer[-i:]):
                return i
        return 0

    @classmethod
    def extract_complete_json_dict(cls, json_str: str):
        try:
            decoder = json.JSONDecoder()
            obj, end_pos = decoder.raw_decode(
                json_str)  # ignore any text after the end of the json object
            if isinstance(obj, dict):
                return obj, end_pos
            return None, 0
        except json.JSONDecodeError:
            return None, 0
