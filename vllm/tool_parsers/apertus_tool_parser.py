# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tool call parser for Apertus models.

Extracts tool calls from the format:
<|tools_prefix|>[{"function_name": {"arg1": "value1", ...}}, ...]<|tools_suffix|>

Used when --enable-auto-tool-choice --tool-call-parser apertus are set.
"""

import json
from collections.abc import Sequence

import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    find_common_prefix,
    partial_json_loads,
)

logger = init_logger(__name__)

# Apertus special tokens for tool calls
TOOL_CALLS_PREFIX = "<|tools_prefix|>"
TOOL_CALLS_SUFFIX = "<|tools_suffix|>"


class ApertusToolParser(ToolParser):
    """
    Tool call parser for Apertus models.

    Handles the extraction of tool calls from text in both non-streaming
    (complete string) and streaming (chunked token) environments.

    The expected Apertus function call format is a JSON array of single-key dictionaries
    sandwiched between special tokens:
    `<|tools_prefix|>[{"function_name": {"arg1": "value1"}}, ...]<|tools_suffix|>`

    Examples:
        >>> tokenizer = ...  # Mock tokenizer
        >>> parser = ApertusToolParser(tokenizer)
        >>> output = 'I will check. <|tools_prefix|>[{"get_weather": '\
            '{"city": "Paris"}}]<|tools_suffix|>'
        >>> request = ChatCompletionRequest(...)
        >>> info = parser.extract_tool_calls(output, request)
        >>> info.content
        "I will check."
        >>> info.tool_calls[0].function.name
        "get_weather"
        >>> info.tool_calls[0].function.arguments
        '{"city": "Paris"}'
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        """
        Initializes the ApertusToolParser.

        Args:
            tokenizer: The model's tokenizer.
                Must be provided to interact with special tokens.
            tools: Optional list of tools available for the current request.

        Raises:
            ValueError: If the `model_tokenizer`
                is not successfully passed to the base class.
        """
        super().__init__(tokenizer, tools)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        # Regex to extract tool calls block (suffix is optional for incomplete outputs)
        self.tool_call_regex = re.compile(
            rf"{re.escape(TOOL_CALLS_PREFIX)}"
            rf"(.*?)"
            rf"(?:{re.escape(TOOL_CALLS_SUFFIX)}|$)",
            re.DOTALL,
        )

        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """
        Resets all streaming state variables for a new completion request.

        This clears the delta text buffer and resets the pointers used to
        track the currently streaming tool index and arguments. Called implicitly
        during initialization and should be called between separate streams.
        """
        self.buffered_delta_text = ""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool: list[str] = []

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """
        Adjusts the generation request to ensure special tool tokens are not skipped.

        Forces `skip_special_tokens=False` if tools are actively being evaluated,
        ensuring the tools special tokens are surfaced to the engine for parsing.

        Args:
            request: The incoming OpenAI-compatible chat completion request.

        Returns:
            The potentially modified chat completion request.
        """
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _buffer_delta_text(self, delta_text: str) -> str:
        """
        Buffers incoming delta chunks to prevent
        fragmentation of multi-token special tags.

        If a chunk ends with a partial match of
        `<|tools_prefix|>` or `<|tools_suffix|>`,
        it holds that part back until the next chunk clarifies if it's the actual tag
        or just normal text.

        Args:
            delta_text: The newly generated text chunk

        Returns:
            The safe, verified text chunk free of partial tag collisions.

        Examples:
            >>> parser = ApertusToolParser(...)
            >>> parser._buffer_delta_text("Let me check <|tool" \
            "Let me check "  # "<|tool" is buffered internally
            >>> parser._buffer_delta_text("s_prefix|>" \
            "<|tools_prefix|>"  # Buffer released on completion
        """
        self.buffered_delta_text += delta_text
        text = self.buffered_delta_text

        for tag in (TOOL_CALLS_PREFIX, TOOL_CALLS_SUFFIX):
            if text.endswith(tag):
                self.buffered_delta_text = ""
                return text

            # Evaluate longest possible partial match first
            for i in range(len(tag) - 1, 0, -1):
                if text.endswith(tag[:i]):
                    self.buffered_delta_text = text[-i:]
                    return text[:-i]

        self.buffered_delta_text = ""
        return text

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extracts tool calls from a completely generated model response (Non-Streaming).

        Args:
            model_output: The full completion string generated by the model.
            request: The current chat completion
                request context containing tool schemas.

        Returns:
            An `ExtractedToolCallInformation` object containing normal text content
            and a list of fully formatted `ToolCall` objects.

        Examples:
            >>> output = 'Let me see. <|tools_prefix|>[{"get_weather":' \
                '{"loc": "Paris"}}]<|tools_suffix|>'
            >>> info = parser.extract_tool_calls(output, request)
            >>> info.tools_called
            True
            >>> info.content
            'Let me see.'
            >>> info.tool_calls[0].function.name
            'get_weather'
        """
        match = self.tool_call_regex.search(model_output)
        if not match:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            # group(1) might contain trailing text if the suffix is missing
            matched_text = match.group(1)
            stripped_text = matched_text.lstrip()

            try:
                # Use raw_decode to robustly isolate
                # the valid JSON array from any trailing garbage
                parsed_json, idx = json.JSONDecoder().raw_decode(stripped_text)
                trailing_in_group = stripped_text[idx:]
            except json.JSONDecodeError:
                # Fallback sequentially to partial parser for token-truncated requests
                parsed_json, _ = partial_json_loads(matched_text, Allow.ALL)
                trailing_in_group = ""

            if not isinstance(parsed_json, list):
                parsed_json = [parsed_json] if parsed_json else []

            tool_calls: list[ToolCall] = []
            for obj in parsed_json:
                if isinstance(obj, dict) and obj:
                    name, args = next(iter(obj.items()))
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            id=make_tool_call_id(),
                            function=FunctionCall(
                                name=name,
                                arguments=json.dumps(args, ensure_ascii=False),
                            ),
                        )
                    )

            # Content combines any generated text
            # prior to and safely after the tool block
            content_str = model_output[: match.start()].strip()

            # Surface any hallucinated text inside
            # the regex group (due to missing suffix)
            if trailing_in_group.strip():
                trailing = trailing_in_group.replace(TOOL_CALLS_SUFFIX, "").strip()
                if trailing:
                    content_str = (content_str + "\n" + trailing).strip()

            # Surface text natively generated after the explicit suffix
            after_suffix = (
                model_output[match.end() :].replace(TOOL_CALLS_SUFFIX, "").strip()
            )
            if after_suffix:
                content_str = (content_str + "\n" + after_suffix).strip()

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_str if content_str else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from Apertus response")
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
        """
        Handles streaming chunks

        Args:
            previous_text: The complete model text generated prior to this chunk.
            current_text: The complete model text including this chunk.
            delta_text: The incremental text addition.
            previous_token_ids: Tokens generated prior to this chunk.
            current_token_ids: Total tokens generated.
            delta_token_ids: Incremental token additions.
            request: The chat completion request.

        Returns:
            A `DeltaMessage` with updated content or tool argument diffs, or `None` if
            the chunk shouldn't emit visible changes yet (e.g. it was purely buffered).

        Examples:
            >>> prev = '<|tools_prefix|>[{"get_weather": {"loc'
            >>> cur = '<|tools_prefix|>[{"get_weather": {"location": "Paris"}}'
            >>> delta = 'ation": "Paris"}}'
            >>> msg = parser.extract_tool_calls_streaming(
            ...     prev, cur, delta, ..., request
            ... )
            >>> msg.tool_calls[0].function.arguments
            'ation": "Paris"}'
        """
        delta_text = self._buffer_delta_text(delta_text)
        if not delta_text:
            return None

        # Fast path: normal text generation before any tools are invoked
        if TOOL_CALLS_PREFIX not in current_text:
            return DeltaMessage(content=delta_text)

        try:
            return self._extract_streaming(current_text, delta_text)
        except Exception:
            logger.exception("Error in Apertus streaming tool call extraction")
            return None

    def _extract_streaming(
        self, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        """
        Core streaming logic.
        Separates visible chat text from JSON blocks and computes diffs.

        Args:
            current_text: The full generated output string so far.
            delta_text: The latest chunk of text added.

        Returns:
            A `DeltaMessage` containing the `content` delta and/or `tool_calls` delta.
        """
        prefix_idx = current_text.rfind(TOOL_CALLS_PREFIX)
        suffix_idx = current_text.rfind(TOOL_CALLS_SUFFIX)

        is_inside_tools = prefix_idx > suffix_idx

        json_completed = False
        json_end_idx: int | None = None

        # Check if the JSON array successfully closed implicitly
        if is_inside_tools:
            json_start = prefix_idx + len(TOOL_CALLS_PREFIX)
            s = current_text[json_start:].lstrip()
            try:
                # If raw_decode succeeds,
                # the JSON array is fully formed and implicitly closed
                _, idx = json.JSONDecoder().raw_decode(s)
                json_end_idx = len(current_text) - len(s) + idx
                json_completed, is_inside_tools = True, False
            except Exception:
                pass

        just_finished = (TOOL_CALLS_SUFFIX in delta_text) or json_completed

        # 1. Fast path: Output normal text immediately
        # if we are completely outside tool block constraints
        if not is_inside_tools and not just_finished:
            text = delta_text.replace(TOOL_CALLS_PREFIX, "").replace(
                TOOL_CALLS_SUFFIX, ""
            )
            return DeltaMessage(content=text) if text else None

        # 2. Extract leading and trailing normal text directly adjacent to tool blocks
        content_str = ""
        if TOOL_CALLS_PREFIX in delta_text:
            content_str += delta_text.split(TOOL_CALLS_PREFIX)[0].replace(
                TOOL_CALLS_SUFFIX, ""
            )

        if just_finished:
            if json_completed and json_end_idx is not None:
                # The tool block finished in this chunk via implicit JSON completion
                # Ensure we strictly isolate
                # and extract only trailing text that is part of `delta_text`
                delta_start_idx = len(current_text) - len(delta_text)
                content_start = max(json_end_idx, delta_start_idx)
                if content_start < len(current_text):
                    content_str += current_text[content_start:].replace(
                        TOOL_CALLS_SUFFIX, ""
                    )
            else:
                content_str += delta_text.split(TOOL_CALLS_SUFFIX)[-1]

        # 3. Extract the isolated JSON array string for the active block
        json_start = prefix_idx + len(TOOL_CALLS_PREFIX)
        json_end = suffix_idx if suffix_idx > prefix_idx else json_end_idx
        json_str = current_text[json_start:json_end]

        tool_calls = self._parse_and_diff_json(json_str, is_final=not is_inside_tools)

        if tool_calls or content_str:
            return DeltaMessage(
                content=content_str if content_str else None,
                tool_calls=tool_calls if tool_calls else None,
            )

        return None

    def _parse_and_diff_json(
        self, json_str: str, is_final: bool
    ) -> list[DeltaToolCall]:
        """
        Parses an isolated, potentially incomplete streaming JSON array and returns
        newly accumulated tool call diffs.

        Args:
            json_str: The extracted JSON array string so far
                (e.g. `[{"weather": {"city": "Par"}]`).
            is_final: True if the tool block has received its closing`<|tools_suffix|>`

        Returns:
            A list of `DeltaToolCall`
            items representing string diffs in function arguments
            to stream back to the client.
        """
        try:
            parsed, _ = partial_json_loads(json_str, Allow.ALL)
            if not isinstance(parsed, list):
                parsed = [parsed] if parsed else []
        except Exception:
            return []

        if not parsed:
            return []

        tool_calls: list[DeltaToolCall] = []
        latest_index = len(parsed) - 1

        # Catch up and finalize any tools we fully skipped over in one large text delta
        while self.current_tool_id < latest_index:
            if self.current_tool_id >= 0:
                if not self.current_tool_name_sent:
                    self._emit_tool_name(parsed, self.current_tool_id, tool_calls)

                delta = self._get_tool_diff(parsed, self.current_tool_id, is_final=True)
                if delta:
                    tool_calls.append(delta)

            self.current_tool_id += 1
            self.current_tool_name_sent = False
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

        # Stream the currently active tool
        if self.current_tool_id >= 0:
            if not self.current_tool_name_sent:
                self._emit_tool_name(parsed, self.current_tool_id, tool_calls)

            delta = self._get_tool_diff(parsed, self.current_tool_id, is_final)
            if delta:
                tool_calls.append(delta)

        return tool_calls

    def _emit_tool_name(
        self, parsed: list, index: int, tool_calls: list[DeltaToolCall]
    ) -> None:
        """
        Extracts and emits the function name mapped to a new tool call ID.

        Args:
            parsed: The partially parsed JSON list containing tool dictionaries.
            index: The active index within the JSON list.
            tool_calls: The running list of delta chunks to mutate.

        Examples:
            Appends `DeltaToolCall(index=0,
                function=DeltaFunctionCall(name="get_weather", ...))`
            to the `tool_calls` list and marks the name as sent.
        """
        obj = parsed[index]
        if isinstance(obj, dict) and obj:
            name = next(iter(obj))
            self.current_tool_name_sent = True
            tool_calls.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(name=name, arguments="").model_dump(
                        exclude_none=True
                    ),
                )
            )

    def _get_tool_diff(
        self, parsed: list, index: int, is_final: bool
    ) -> DeltaToolCall | None:
        """
        Calculates the exact string difference to safely append new tool parameters.

        This ensures characters like `{`, `}`, and `"` don't jump around unevenly
        in the UI frontend while streaming incomplete JSON arguments.

        Args:
            parsed: The latest list of parsed JSON objects.
            index: The active tool's array index.
            is_final: Whether to emit
                trailing structural brackets (True if block is done).

        Returns:
            A `DeltaToolCall` mapping to the arguments diff,
                or None if no text was appended.

        Examples:
            >>> # Previous streamed state: '{"city": "Pari'
            >>> # Current full parse state: '{"city": "Paris"}'
            >>> # Returns diff (closing bracket suppressed until final):
            >>> parser._get_tool_diff(parsed, index=0, is_final=False)
            DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='s'))
        """
        obj = parsed[index]
        if not isinstance(obj, dict) or not obj:
            return None

        name, args = next(iter(obj.items()))
        if args is None:
            return None

        args_json = json.dumps(args, ensure_ascii=False)

        # Suppress trailing structural characters
        # during stream (looks cleaner in frontends)
        if not is_final:
            while args_json and args_json[-1] in ("}", '"', "]", " ", ","):
                args_json = args_json[:-1]

        prev_sent = self.streamed_args_for_tool[index]
        if args_json == prev_sent:
            return None

        prefix = find_common_prefix(prev_sent, args_json)
        if len(prefix) < len(prev_sent):
            # Backtrack state if partial parser structurally updates a past assumption
            self.streamed_args_for_tool[index] = prefix
            return None

        diff = args_json[len(prev_sent) :]
        if diff:
            self.streamed_args_for_tool[index] = args_json
            return DeltaToolCall(
                index=index,
                function=DeltaFunctionCall(arguments=diff).model_dump(
                    exclude_none=True
                ),
            )

        return None
