# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mistral tool call parser for v11+ models.

This implementation uses token-based parsing for streaming, leveraging the
atomic nature of special token IDs ([TOOL_CALLS], [ARGS], [CALL_ID]) to
reliably detect tool call boundaries.

Supported models: Mistral-Small-3.1+, Ministral-3+, and other v11+ models.

Note: Pre-v11 models (Mistral-7B-Instruct-v0.1/v0.2/v0.3) are not supported.
These older models have limited tool calling capabilities and require complex
text-based parsing with partial JSON handling. Users should upgrade to v11+
models for reliable tool calling support.
"""

import json
from collections.abc import Sequence
from enum import Enum, auto
from random import choices
from string import ascii_letters, digits

import regex as re
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
    PARSING_TOOL_NAME = auto()  # After [TOOL_CALLS], parsing function name
    PARSING_TOOL_ARGS = auto()  # Parsing JSON arguments
    COMPLETE = auto()  # All tools parsed


class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral v11+ models.

    Supports the v11+ format: [TOOL_CALLS]name[ARGS]{...}
    Optionally with call ID: [TOOL_CALLS]name[CALL_ID]id[ARGS]{...}

    This parser requires MistralTokenizer (tokenizer_mode=mistral) and
    models using tokenizer version 11 or higher.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        if not isinstance(self.model_tokenizer, MistralTokenizer):
            raise RuntimeError(
                "MistralToolParser requires MistralTokenizer. "
                "Please use tokenizer_mode='mistral' in your vLLM configuration. "
                "Note: Only v11+ Mistral models are supported for tool calling."
            )

        self._mistral_base_tokenizer = self.model_tokenizer.tokenizer
        self._version = self.model_tokenizer.version

        if self._version < 11:
            raise RuntimeError(
                f"MistralToolParser requires tokenizer version 11 or higher, "
                f"but got version {self._version}. Pre-v11 models "
                "(Mistral-7B-Instruct-v0.1/v0.2/v0.3) are not supported for "
                "tool calling. Please use a v11+ model such as "
                "Mistral-Small-3.1 or Ministral-3."
            )

        # Get bot token info
        self.bot_token = "[TOOL_CALLS]"
        self.bot_token_id = self.vocab.get(self.bot_token)

        if self.bot_token_id is None:
            raise RuntimeError(
                "Mistral Tool Parser could not locate the [TOOL_CALLS] token "
                "in the tokenizer!"
            )

        # Get control tokens for v11+ format
        try:
            self._args_token_id = self._mistral_base_tokenizer.get_control_token(
                "[ARGS]"
            )
        except Exception:
            raise RuntimeError(
                "Mistral Tool Parser could not locate the [ARGS] token. "
                "This token is required for v11+ tool call parsing."
            )

        self._call_id_token_id: int | None = None
        try:
            self._call_id_token_id = self._mistral_base_tokenizer.get_control_token(
                "[CALL_ID]"
            )
        except Exception:
            # [CALL_ID] is optional - some models may not have it
            pass

        # Regex for non-streaming parsing: name{args}
        self.fn_name_regex = re.compile(
            r"([a-zA-Z0-9_-]+)(\{[\s\S]*?\}+)", re.DOTALL
        )

        # Streaming state
        self._streaming_state = StreamingState.CONTENT
        self._current_tool_index = -1
        self._current_tool_id: str | None = None
        self._current_tool_name: str = ""
        self._current_tool_args: str = ""
        self._brace_depth = 0

        # For compatibility with serving_chat.py's finish_reason detection
        self.prev_tool_call_arr: list[dict] = []

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model response.

        Parses the v11+ format: [TOOL_CALLS]name{args}[TOOL_CALLS]name{args}...
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

            # Parse tool calls from each segment after [TOOL_CALLS]
            function_call_arr = []
            for segment in model_output.split(self.bot_token):
                if not segment.strip():
                    continue

                matches = self.fn_name_regex.findall(segment)
                for match in matches:
                    fn_name = match[0]
                    args = match[1]
                    function_call_arr.append(
                        {"name": fn_name, "arguments": json.loads(args)}
                    )

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
        Extract tool calls from streaming output using token-based parsing.

        Token IDs are atomic - they cannot be split across chunks - which
        eliminates a whole class of parsing bugs that affect text-based parsing.
        """
        # If no tool call token seen yet, emit as content
        if self.bot_token_id not in current_token_ids:
            return DeltaMessage(content=delta_text)

        return self._stream_tool_calls(delta_token_ids)

    def _stream_tool_calls(
        self, delta_token_ids: Sequence[int]
    ) -> DeltaMessage | None:
        """
        Stream tool calls using token-based parsing.

        Detects [TOOL_CALLS] and [ARGS] tokens to identify tool call boundaries,
        then streams function names and arguments as they arrive.
        """
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        delta_tool_calls: list[DeltaToolCall] = []

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
                # Before any tool call - shouldn't happen if bot_token_id
                # is in current_token_ids, but handle gracefully
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
                    self.streamed_args_for_tool[self._current_tool_index] = (
                        self._current_tool_args
                    )

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
