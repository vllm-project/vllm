# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mistral tool call parser supporting both streaming and non-streaming modes.

This implementation:
1. Non-streaming: Uses string/regex-based parsing (text is already decoded)
2. Streaming: Uses token-based parsing for v11+ (token IDs are atomic)
3. Supports all tokenizer versions (v2-v13)
"""

import json
from collections.abc import Sequence
from enum import Enum, auto
from random import choices
from string import ascii_letters, digits

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow
from pydantic import Field

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
from vllm.logger import init_logger
from vllm.tokenizers import MistralTokenizer, TokenizerLike

logger = init_logger(__name__)

ALPHANUMERIC = ascii_letters + digits


class MistralToolCall(ToolCall):
    id: str = Field(default_factory=lambda: MistralToolCall.generate_random_id())

    @staticmethod
    def generate_random_id():
        # Mistral Tool Call Ids must be alphanumeric with a length of 9.
        # https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L299
        return "".join(choices(ALPHANUMERIC, k=9))

    @staticmethod
    def is_valid_id(id: str) -> bool:
        return id.isalnum() and len(id) == 9


class StreamingState(Enum):
    """Streaming state for tool call parsing."""

    CONTENT = auto()  # Before any [TOOL_CALLS] token
    PARSING_TOOL_NAME = auto()  # After [TOOL_CALLS], parsing function name (v11+)
    PARSING_TOOL_ARGS = auto()  # Parsing JSON arguments
    COMPLETE = auto()  # All tools parsed


class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral models.

    Supports two formats:
    - Pre-v11: [TOOL_CALLS][{"name": "...", "arguments": {...}}, ...]
    - V11+: [TOOL_CALLS]name[ARGS]{...} or [TOOL_CALLS]name[CALL_ID]id[ARGS]{...}
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self._use_mistral_tokenizer = isinstance(self.model_tokenizer, MistralTokenizer)

        if not self._use_mistral_tokenizer:
            logger.info(
                "Non-Mistral tokenizer detected when using a Mistral model..."
            )

        # Get bot token info
        self.bot_token = "[TOOL_CALLS]"
        self.bot_token_id = self.vocab.get(self.bot_token)

        if self.bot_token_id is None:
            raise RuntimeError(
                "Mistral Tool Parser could not locate the tool call token in "
                "the tokenizer!"
            )

        # For MistralTokenizer, get additional info for v11+ parsing
        self._args_token_id: int | None = None
        self._call_id_token_id: int | None = None
        self._version: int = 3  # Default to pre-v11

        if self._use_mistral_tokenizer:
            assert isinstance(self.model_tokenizer, MistralTokenizer)
            self._mistral_base_tokenizer = self.model_tokenizer.tokenizer
            self._version = self.model_tokenizer.version

            # Get control tokens for v11+ format
            if self._version >= 11:
                try:
                    self._args_token_id = self._mistral_base_tokenizer.get_control_token(
                        "[ARGS]"
                    )
                except Exception:
                    pass
                try:
                    self._call_id_token_id = (
                        self._mistral_base_tokenizer.get_control_token("[CALL_ID]")
                    )
                except Exception:
                    pass

        # Regex patterns
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)
        # V11+ format: name{args} where name is alphanumeric with underscores/hyphens
        self.fn_name_regex = re.compile(
            r"([a-zA-Z0-9_-]+)(\{[\s\S]*?\}+)", re.DOTALL
        ) if self._is_v11_plus() else None

        # Streaming state
        self._streaming_state = StreamingState.CONTENT
        self._current_tool_index = -1
        self._current_tool_id: str | None = None
        self._current_tool_name: str = ""
        self._current_tool_args: str = ""
        self._accumulated_tokens: list[int] = []
        self._brace_depth = 0

        # For compatibility with serving_chat.py's finish_reason detection
        self.prev_tool_call_arr: list[dict] = []

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if (
            not self._use_mistral_tokenizer
            and request.tools
            and request.tool_choice != "none"
        ):
            # Do not skip special tokens when using chat template
            # with Mistral parser as TOOL_CALL token is needed
            # for tool detection.
            # Note: we don't want skip_special_tokens=False
            # with MistralTokenizer as it is incompatible
            request.skip_special_tokens = False
        return request

    def _is_v11_plus(self) -> bool:
        """Check if using v11+ tokenizer format."""
        return self._use_mistral_tokenizer and self._version >= 11

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model response.

        Uses string-based parsing since model_output is already decoded text.
        """
        # Fast path: no tool call token present
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            # Get content before tool calls
            content = model_output.split(self.bot_token)[0]
            content = content if content.strip() else None

            # Remove bot tokens and parse tool calls
            tool_content = model_output.replace(self.bot_token, "").strip()

            if self._is_v11_plus():
                # V11+ format: name{args} repeated for each tool call
                function_call_arr = self._parse_v11_tool_calls(model_output)
            else:
                # Pre-v11 format: JSON array
                function_call_arr = self._parse_pre_v11_tool_calls(tool_content)

            # Convert to MistralToolCall objects
            tool_calls: list[MistralToolCall] = [
                MistralToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        arguments=json.dumps(
                            raw_function_call["arguments"], ensure_ascii=False
                        ),
                    ),
                )
                for raw_function_call in function_call_arr
            ]

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output.replace(self.bot_token, "").strip(),
            )

    def _parse_v11_tool_calls(self, model_output: str) -> list[dict]:
        """Parse v11+ format: [TOOL_CALLS]name{args}[TOOL_CALLS]name{args}..."""
        function_call_arr = []

        # Split by [TOOL_CALLS] and process each segment
        for segment in model_output.split(self.bot_token):
            if not segment.strip():
                continue

            assert self.fn_name_regex is not None
            matches = self.fn_name_regex.findall(segment)

            for match in matches:
                fn_name = match[0]
                args = match[1]
                function_call_arr.append(
                    {"name": fn_name, "arguments": json.loads(args)}
                )

        return function_call_arr

    def _parse_pre_v11_tool_calls(self, tool_content: str) -> list[dict]:
        """Parse pre-v11 format: [{"name": "...", "arguments": {...}}, ...]"""
        try:
            return json.loads(tool_content)
        except json.JSONDecodeError:
            # Try regex extraction as fallback
            raw_tool_call = self.tool_call_regex.findall(tool_content)[0]
            return json.loads(raw_tool_call)

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
        Extract tool calls from streaming output.

        For v11+ with MistralTokenizer, uses token-based parsing.
        Otherwise, uses text-based partial JSON parsing.
        """
        # If no tool call token seen yet, emit as content
        if self.bot_token_id not in current_token_ids:
            return DeltaMessage(content=delta_text)

        # Accumulate tokens for parsing
        self._accumulated_tokens.extend(delta_token_ids)

        # Route to appropriate streaming parser
        if self._is_v11_plus():
            return self._stream_v11_plus(delta_token_ids, delta_text)
        else:
            return self._stream_pre_v11(delta_text, current_text)

    def _stream_v11_plus(
        self, delta_token_ids: Sequence[int], delta_text: str
    ) -> DeltaMessage | None:
        """
        Stream tool calls for v11+ format: [TOOL_CALLS]name[ARGS]{...}

        Uses token-based parsing since token IDs are atomic.
        """
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        delta_tool_calls: list[DeltaToolCall] = []
        content_delta: str | None = None

        for token_id in delta_token_ids:
            if token_id == self.bot_token_id:
                # Starting a new tool call
                self._current_tool_index += 1
                self._current_tool_id = MistralToolCall.generate_random_id()
                self._current_tool_name = ""
                self._current_tool_args = ""
                self._brace_depth = 0
                self._streaming_state = StreamingState.PARSING_TOOL_NAME

                # Set flag for finish_reason detection
                if not self.prev_tool_call_arr:
                    self.prev_tool_call_arr = [{"arguments": {}}]

                # Initialize streamed_args_for_tool for this tool index
                while len(self.streamed_args_for_tool) <= self._current_tool_index:
                    self.streamed_args_for_tool.append("")

            elif token_id == self._args_token_id:
                # Transition from name to arguments
                if self._streaming_state == StreamingState.PARSING_TOOL_NAME:
                    # Emit the complete function name
                    delta_tool_calls.append(
                        DeltaToolCall(
                            index=self._current_tool_index,
                            type="function",
                            id=self._current_tool_id,
                            function=DeltaFunctionCall(
                                name=self._current_tool_name.strip()
                            ).model_dump(exclude_none=True),
                        )
                    )
                    self._streaming_state = StreamingState.PARSING_TOOL_ARGS

            elif token_id == self._call_id_token_id:
                # Skip call ID tokens (they come between name and [ARGS])
                # We generate our own IDs
                pass

            elif self._streaming_state == StreamingState.CONTENT:
                # Before any tool call - this shouldn't happen if bot_token_id is in current_token_ids
                # but handle it gracefully
                pass

            elif self._streaming_state == StreamingState.PARSING_TOOL_NAME:
                # Accumulate name tokens
                token_str = self._mistral_base_tokenizer.decode(
                    [token_id], special_token_policy=SpecialTokenPolicy.IGNORE
                )
                self._current_tool_name += token_str

            elif self._streaming_state == StreamingState.PARSING_TOOL_ARGS:
                # Stream argument tokens
                token_str = self._mistral_base_tokenizer.decode(
                    [token_id], special_token_policy=SpecialTokenPolicy.IGNORE
                )

                # Track brace depth for nested JSON
                for char in token_str:
                    if char == "{":
                        self._brace_depth += 1
                    elif char == "}":
                        self._brace_depth -= 1

                self._current_tool_args += token_str

                # Update streamed_args_for_tool for vLLM's finish handling
                if self._current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self._current_tool_index] = self._current_tool_args

                # Emit arguments delta
                delta_tool_calls.append(
                    DeltaToolCall(
                        index=self._current_tool_index,
                        function=DeltaFunctionCall(
                            arguments=token_str
                        ).model_dump(exclude_none=True),
                    )
                )

        # Build response
        if delta_tool_calls:
            return DeltaMessage(tool_calls=delta_tool_calls)

        return None

    def _stream_pre_v11(
        self, delta_text: str, current_text: str
    ) -> DeltaMessage | None:
        """
        Stream tool calls for pre-v11 format: [TOOL_CALLS][{...}, {...}]

        Uses partial JSON parsing for structure, but streams raw text for arguments.
        """
        # Parse tool calls from text after [TOOL_CALLS]
        if self.bot_token not in current_text:
            return DeltaMessage(content=delta_text)

        # Handle bot token in delta - but only if it's just the token arriving
        # (not a one-chunk case where everything arrives at once)
        if self.bot_token in delta_text:
            if not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]
            # Check if this is just the bot token arriving (not one-chunk)
            parts = delta_text.split(self.bot_token)
            content_before = parts[0]
            content_after = parts[-1].strip() if len(parts) > 1 else ""

            # If there's significant content after the bot token, this is a one-chunk case
            # Continue processing instead of returning early
            if content_before and not content_after:
                return DeltaMessage(content=content_before)
            elif not content_after:
                # Just the bot token, nothing more - wait for more tokens
                return None
            # Otherwise fall through to process the full content

        parsable = current_text.split(self.bot_token)[-1]

        # Partial JSON parsing to detect tool call structure
        flags = Allow.ALL if self._current_tool_name else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr: list[dict] = partial_json_parser.loads(parsable, flags)
        except Exception:
            return None

        if not tool_call_arr:
            return None

        delta_tool_calls: list[DeltaToolCall] = []

        # Check if we've moved to a new tool
        if len(tool_call_arr) > self._current_tool_index + 1:
            self._current_tool_index = len(tool_call_arr) - 1
            self._current_tool_name = ""
            self._current_tool_args = ""
            self._current_tool_id = MistralToolCall.generate_random_id()

            # Initialize streamed_args_for_tool for this tool index
            while len(self.streamed_args_for_tool) <= self._current_tool_index:
                self.streamed_args_for_tool.append("")

        current_tool = tool_call_arr[self._current_tool_index]

        # Emit function name if available and not yet emitted
        if "name" in current_tool and not self._current_tool_name:
            self._current_tool_name = current_tool["name"]
            delta_tool_calls.append(
                DeltaToolCall(
                    index=self._current_tool_index,
                    type="function",
                    id=self._current_tool_id,
                    function=DeltaFunctionCall(
                        name=self._current_tool_name
                    ).model_dump(exclude_none=True),
                )
            )

        # Emit arguments delta using raw text extraction
        if "arguments" in current_tool and self._current_tool_name:
            # Find the current tool's arguments in the parsable text.
            # For multiple tools, we need to find the N-th "arguments" occurrence.
            # Use findall to get all matches and pick the right one.
            matches = list(re.finditer(
                r'"arguments"\s*:\s*(\{)',
                parsable
            ))

            if matches and len(matches) > self._current_tool_index:
                # Get the start position for current tool's arguments
                match = matches[self._current_tool_index]
                args_start = match.start(1)  # Start of the '{'

                # Extract from args_start to end, then trim to the arguments object
                raw_args = parsable[args_start:]

                # Trim trailing characters that aren't part of this tool's arguments.
                # The raw_args may include `}}]` or `}, {...` for the next tool.
                # We track brace depth to find where the arguments object ends.
                brace_depth = 0
                end_pos = len(raw_args)
                for i, char in enumerate(raw_args):
                    if char == '{':
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1
                        if brace_depth == 0:
                            # Found the end of the arguments object
                            end_pos = i + 1
                            break

                raw_args = raw_args[:end_pos]

                # Find how much of this raw text is new
                if len(raw_args) > len(self._current_tool_args):
                    args_delta = raw_args[len(self._current_tool_args):]

                    if args_delta:
                        delta_tool_calls.append(
                            DeltaToolCall(
                                index=self._current_tool_index,
                                function=DeltaFunctionCall(
                                    arguments=args_delta
                                ).model_dump(exclude_none=True),
                            )
                        )
                    self._current_tool_args = raw_args

                    # Update streamed_args_for_tool for vLLM's finish handling
                    while len(self.streamed_args_for_tool) <= self._current_tool_index:
                        self.streamed_args_for_tool.append("")
                    self.streamed_args_for_tool[self._current_tool_index] = self._current_tool_args

        if delta_tool_calls:
            return DeltaMessage(tool_calls=delta_tool_calls)

        return None
