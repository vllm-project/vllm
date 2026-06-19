# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

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
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import _extract_tool_info, partial_tag_overlap

logger = init_logger(__name__)


class KimiK2ToolParser(ToolParser):
    structural_tag_model = "kimi"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        # Streaming state
        self._sent_content_idx: int = 0
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        # Section marker
        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"

        # Individual tool call markers
        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_arg_token: str = "<|tool_call_argument_begin|>"

        # Regex for non-streaming extraction
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^\s<|]+)\s*"
            r"<\|tool_call_argument_begin\|>\s*"
            r"(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*"
            r"<\|tool_call_end\|>",
            re.DOTALL,
        )
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure special-token markers appear as literal text in
            # current_text so we can do pure text-based parsing.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        else:
            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = self.tool_call_regex.findall(model_output)

                logger.debug("function_call_tuples: %s", function_call_tuples)

                tool_calls = []
                for match in function_call_tuples:
                    function_id, function_args = match
                    tool_id, function_name = self._extract_tool_id_and_name(
                        function_id, function_args
                    )
                    if not tool_id or not function_name:
                        continue
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            type="function",
                            function=FunctionCall(
                                name=function_name, arguments=function_args
                            ),
                        )
                    )

                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=len(tool_calls) > 0,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent content before the tool-calls section, or None.

        Holds back any trailing suffix that partially matches
        ``<|tool_calls_section_begin|>`` to avoid leaking marker bytes.
        """
        if self.tool_calls_start_token not in current_text:
            overlap = partial_tag_overlap(current_text, self.tool_calls_start_token)
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = current_text.index(self.tool_calls_start_token)

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_calls(self, current_text: str) -> list[str]:
        """Extract raw bodies from ``<|tool_call_begin|>…<|tool_call_end|>`` blocks."""
        if self.tool_calls_start_token not in current_text:
            return []

        results: list[str] = []
        pos = current_text.index(self.tool_calls_start_token)
        while True:
            start = current_text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            tc_start = start + len(self.tool_call_start_token)
            end = current_text.find(self.tool_call_end_token, tc_start)

            if end != -1:
                tool_call = current_text[tc_start:end]
                pos = end + len(self.tool_call_end_token)
            else:
                tool_call = current_text[tc_start:]
                overlap = partial_tag_overlap(tool_call, self.tool_call_end_token)
                if overlap:
                    tool_call = tool_call[:-overlap]

            results.append(tool_call)

            if end == -1:
                break
        return results

    def _extract_tool_id_and_name(
        self,
        header: str | None,
        tool_args: str | None,
    ) -> tuple[str | None, str | None]:
        """Parse ``(tool_id, tool_name)`` from a header
        like ``"functions.get_weather:0"``.

        Kimi K2.5 may emit a bare numeric counter instead of the tool name. In
        that case, preserve the native ID and infer the tool name when the
        request tools make it unambiguous.
        """
        if header is None:
            return None, None
        tool_id = header.strip()
        if tool_id.isdigit():
            return tool_id, self._infer_tool_name(tool_args)

        tool_name, sep, tool_index = tool_id.rpartition(":")
        if sep and tool_name and tool_index.isdigit():
            return tool_id, tool_name.split(".")[-1]
        return None, None

    def _infer_tool_name(self, tool_args: str | None) -> str | None:
        if not self.tools:
            return None
        if len(self.tools) == 1:
            return _extract_tool_info(self.tools[0])[0]
        if not tool_args:
            return None

        try:
            parsed_args = json.loads(tool_args)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed_args, dict) or not parsed_args:
            return None

        matching_name = None
        arg_keys = set(parsed_args)
        for tool in self.tools:
            tool_name, parameters = _extract_tool_info(tool)
            properties = (parameters or {}).get("properties", {})
            if not isinstance(properties, dict) or not arg_keys <= properties.keys():
                continue
            if matching_name is not None:
                return None
            matching_name = tool_name
        return matching_name

    def _split_tool_call(self, tool_call: str) -> tuple[str | None, str | None]:
        """Split a tool-call body into ``(header, arguments)`` at the argument marker.

        Example::
            'get_weather:0 <|tool_call_argument_begin|>{"c'
            -> ("get_weather:0", '{"c')
        """
        arg_pos = tool_call.find(self.tool_call_arg_token)
        if arg_pos == -1:
            return None, None
        header = tool_call[:arg_pos].strip()
        tool_args = tool_call[arg_pos + len(self.tool_call_arg_token) :]
        return header, tool_args

    def _compute_args_diff(self, index: int, tool_args: str | None) -> str | None:
        """Return new argument text not yet sent for tool `index`, or None."""
        if tool_args is None:
            return None
        prev = self.streamed_args_for_tool[index]
        if len(tool_args) <= len(prev):
            return None
        diff = tool_args[len(prev) :]
        self.streamed_args_for_tool[index] = tool_args
        self.prev_tool_call_arr[index]["arguments"] = tool_args
        return diff

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
        try:
            # Extract any content before tool calls.
            content = self._extract_content(current_text)
            tool_calls = self._extract_tool_calls(current_text)
            tool_call_deltas: list[DeltaToolCall] = []

            for i, tool_call in enumerate(tool_calls):
                # First time seeing tool call at index i.
                if i >= len(self.prev_tool_call_arr):
                    # Initialize streaming state.
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                header, tool_args = self._split_tool_call(tool_call)

                # Stream back tool name.
                if "name" not in self.prev_tool_call_arr[i]:
                    tool_id, tool_name = self._extract_tool_id_and_name(
                        header, tool_args
                    )
                    if not tool_name:
                        # Can't skip to tool i+1 if i isn't ready
                        break
                    self.prev_tool_call_arr[i]["name"] = tool_name
                    self.prev_tool_call_arr[i]["id"] = tool_id
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=tool_id,
                            function=DeltaFunctionCall(name=tool_name).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                # Stream back new tool args by diffing against what was sent.
                args_diff = self._compute_args_diff(i, tool_args)
                if args_diff:
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=args_diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

            if content or tool_call_deltas:
                return DeltaMessage(
                    content=content,
                    tool_calls=tool_call_deltas,
                )
            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
