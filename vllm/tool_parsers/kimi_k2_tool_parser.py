# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
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
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)
from vllm.tool_parsers.utils import partial_tag_overlap

logger = init_logger(__name__)


class KimiK2ToolParser(ToolParser):
    supports_required_and_named = False

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
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*"
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
        structure_tag = None
        chat_request = None
        if (
            isinstance(request, ChatCompletionRequest)
            and request.tools
            and (
                request.tool_choice == "required"
                or isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
            )
        ):
            chat_request = request
            structure_tag = self.get_structural_tag(chat_request)

        if structure_tag is None:
            request = super().adjust_request(request)
        else:
            structural_tag = json.dumps(structure_tag.model_dump())
            assert chat_request is not None
            if chat_request.structured_outputs is not None:
                chat_request.structured_outputs = StructuredOutputsParams(
                    structural_tag=structural_tag,
                    disable_any_whitespace=(
                        chat_request.structured_outputs.disable_any_whitespace
                    ),
                    disable_additional_properties=(
                        chat_request.structured_outputs.disable_additional_properties
                    ),
                    whitespace_pattern=chat_request.structured_outputs.whitespace_pattern,
                )
            else:
                chat_request.structured_outputs = StructuredOutputsParams(
                    structural_tag=structural_tag
                )
            chat_request.response_format = None
            request = chat_request

        if request.tools and request.tool_choice != "none":
            # Ensure special-token markers appear as literal text in
            # current_text so we can do pure text-based parsing.
            request.skip_special_tokens = False
        return request

    def get_structural_tag(self, request: ChatCompletionRequest):
        chat_template_kwargs = request.chat_template_kwargs or {}
        thinking = bool(chat_template_kwargs.get("thinking", True))
        reasoning = get_enable_structured_outputs_in_reasoning() and thinking
        structural_tag = get_model_structural_tag(
            model="kimi_k2",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=reasoning,
        )
        # Parser-owned grammar from the first generated token, but only when the
        # tag has no reasoning prefix (``reasoning`` is False). In that case the
        # engine would otherwise defer the bitmask during reasoning and the
        # forced tool would not be enforced from token 0, so we set
        # ``reasoning_ended=True`` here. When the tag *does* carry a
        # ``<think>...</think>`` prefix (reasoning on), ``enable_in_reasoning``
        # already drives the grammar from token 0; leaving this flag unset lets
        # the reasoning parser extract the reasoning content (setting it would
        # suppress extraction and leak ``</think>`` into ``content``).
        request._grammar_from_tool_parser = not reasoning
        return structural_tag

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
                    # function_id: functions.get_weather:0 or get_weather:0
                    function_name = function_id.split(":")[0].split(".")[-1]
                    tool_calls.append(
                        ToolCall(
                            id=function_id,
                            type="function",
                            function=FunctionCall(
                                name=function_name, arguments=function_args
                            ),
                        )
                    )

                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
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

    @staticmethod
    def _extract_tool_id_and_name(
        header: str | None,
    ) -> tuple[str | None, str | None]:
        """Parse ``(tool_id, tool_name)`` from a header
        like ``"functions.get_weather:0"``."""
        if header is None:
            return None, None
        match = re.match(r"(.+:\d+)", header)
        if not match:
            return None, None

        tool_id = match.group(1).strip()
        tool_name = tool_id.split(":")[0].split(".")[-1]
        return tool_id, tool_name

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
                    tool_id, tool_name = self._extract_tool_id_and_name(header)
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
