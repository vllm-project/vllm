# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the HyperCLOVA X vLLM plugin (NAVER Cloud, Apache-2.0).

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.utils import Tool

logger = init_logger(__name__)


class HyperCLOVAXSeedThink14BToolParser(ToolParser):
    """Tool-call parser for ``HyperCLOVAX-SEED-Think-14B``.

    The 14B model emits tool calls as a JSON array following the
    ``<|im_start|>assistant -> tool/function_call`` delimiter. This parser
    handles that header form, the bare-array hand-off used after the reasoning
    parser strips the header, and ``arguments``/``parameters`` aliasing.
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.tool_call_start_token: str = " -> tool/function_call\n"
        self.tool_call_end_token: str = "<|im_end|>"
        # Non-greedy ``.*?`` with an explicit end-of-string anchor avoids
        # matching a ``]`` that appears inside a string value in arguments.
        self.tool_call_regex = re.compile(
            r"-> tool/function_call\n(.*?)<\|im_end\|>"
            r"|-> tool/function_call\n(.*?)\](\s*)$",
            re.DOTALL,
        )

        # for streaming
        self.tool_call_offset = 0
        self.current_tool_id = -1
        # vLLM serving references these incremental snapshots while building
        # streamed tool-call chunks, so keep them in sync even if this parser
        # does not read them directly.
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool: list[str] = []
        self.is_reasoning_ended = False
        self.is_bare_tool_call_mode = False
        self.pending_bare_tool_call_prefix = ""
        self.bare_tool_call_was_reasoning_ended = False
        self.has_emitted_reasoning_text = False

        self.buffer_string = ""
        self.special_strings = [
            "<|im_end|>\n",
            "<|im_end|>",
            "<|im_start|>assistant/think\n",
            "<|im_start|>assistant\n",
            "-> tool/function_call\n",
            "<|stop|>",
            "<|endofturn|>",
        ]
        self.escaped_special_strings = [re.escape(ss) for ss in self.special_strings]

    def _delta(
        self, *, reasoning: str | None = None, content: str | None = None
    ) -> DeltaMessage:
        # Use ``is not None`` so empty strings are not silently discarded.
        kwargs = {}
        if reasoning is not None:
            kwargs["reasoning"] = reasoning
        if content is not None:
            kwargs["content"] = content
        return DeltaMessage(**kwargs)

    # depth + string-aware JSON object boundary scanner
    # replaces the naive brace-position scanning approach
    def _find_json_object_end(self, text: str, start: int) -> int | None:
        """Returns the index of the closing } that ends the JSON object
        starting at text[start]. Returns None if the object is incomplete."""
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
        return None

    def _normalize_function_call(self, function_call: dict) -> dict:
        return {
            "name": function_call.get("name", ""),
            "arguments": function_call.get(
                "arguments", function_call.get("parameters", {})
            ),
        }

    def _arguments_json(self, function_call: dict) -> str:
        return json.dumps(function_call.get("arguments", {}), ensure_ascii=False)

    def _tool_names(self, request: ChatCompletionRequest) -> set[str]:
        tool_names: set[str] = set()
        for tool in getattr(request, "tools", None) or []:
            if isinstance(tool, dict):
                function = tool.get("function") or {}
                name = function.get("name") if isinstance(function, dict) else None
            else:
                function = getattr(tool, "function", None)
                name = getattr(function, "name", None)
            if name:
                tool_names.add(name)
        return tool_names

    def _request_skips_reasoning(self, request: ChatCompletionRequest) -> bool:
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
        return bool(chat_template_kwargs.get("skip_reasoning", False)) and not bool(
            chat_template_kwargs.get("force_reasoning", False)
        )

    def _request_forces_reasoning(self, request: ChatCompletionRequest) -> bool:
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
        return bool(chat_template_kwargs.get("force_reasoning", False))

    def _is_function_call_like(
        self,
        function_call: object,
        request: ChatCompletionRequest,
    ) -> bool:
        if not isinstance(function_call, dict):
            return False

        name = function_call.get("name")
        if not isinstance(name, str) or not name:
            return False

        if "arguments" not in function_call and "parameters" not in function_call:
            return False

        tool_names = self._tool_names(request)
        return not tool_names or name in tool_names

    def _is_partial_tool_role_fragment(self, text: str) -> bool:
        normalized = text.lstrip()
        target = self.tool_call_start_token.lstrip()
        return (
            bool(normalized) and normalized != target and target.startswith(normalized)
        )

    def _is_tool_role_prefix_or_fragment(self, text: str) -> bool:
        normalized = text.lstrip()
        target = self.tool_call_start_token.lstrip()
        return bool(normalized) and target.startswith(normalized)

    def _emit_text_delta(self, text: str | None) -> DeltaMessage | None:
        if not text:
            return None
        if self.is_reasoning_ended:
            return DeltaMessage(content=text)
        return self._delta(reasoning=text)

    def _flush_buffered_delta(self) -> DeltaMessage | None:
        if not self.buffer_string:
            return None

        flush_text = self.buffer_string
        self.buffer_string = ""
        for special_string in self.special_strings:
            flush_text = flush_text.replace(special_string, "")
        if not flush_text or self._is_tool_role_prefix_or_fragment(flush_text):
            return None
        if not self.is_reasoning_ended and flush_text.strip():
            self.has_emitted_reasoning_text = True
        return self._emit_text_delta(flush_text)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            # Bare array mode: reasoning parser stripped the header before handoff
            stripped = model_output.lstrip("\n")
            if stripped.startswith("[") and getattr(request, "tools", None):
                try:
                    bare_json = stripped
                    end_idx = bare_json.find(self.tool_call_end_token)
                    if end_idx >= 0:
                        bare_json = bare_json[:end_idx]
                    bare_json = bare_json.rstrip()
                    raw_function_calls = json.loads(bare_json)
                    if not all(
                        self._is_function_call_like(fc, request)
                        for fc in raw_function_calls
                    ):
                        raise ValueError("Bare array is not a tool-call array.")
                    tool_calls = [
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=fc["name"],
                                arguments=self._arguments_json(
                                    self._normalize_function_call(fc)
                                ),
                            ),
                        )
                        for fc in raw_function_calls
                    ]
                    return ExtractedToolCallInformation(
                        tools_called=True, tool_calls=tool_calls, content=None
                    )
                except Exception:
                    logger.debug(
                        "Bare array tool call parsing failed. raw=%s",
                        model_output[:200],
                    )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # guard against no regex match before using raw_function_calls
        tool_call_match = self.tool_call_regex.search(model_output)
        if not tool_call_match:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            if tool_call_match.group(1) is not None:
                raw_function_calls = json.loads(tool_call_match.group(1))
            else:
                raw_function_calls = json.loads(tool_call_match.group(2) + "]")

            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_call["name"],
                        arguments=self._arguments_json(
                            self._normalize_function_call(function_call)
                        ),
                    ),
                )
                for function_call in raw_function_calls
            ]

            # check if there is other content before tool calls
            if (
                "<|im_end|>\n<|im_start|>assistant -> tool/function_call\n"
                in model_output
            ):
                content = model_output.split(
                    "<|im_end|>\n<|im_start|>assistant -> tool/function_call\n"
                )[0]

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )
            else:
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=None
                )

        # split exception handling for better diagnostics
        except json.JSONDecodeError:
            logger.exception(
                "Failed to decode tool call JSON. raw=%s", model_output[:300]
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        except KeyError as e:
            logger.exception(
                "Missing required field in tool call: %s. raw=%s", e, model_output[:300]
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        except Exception:
            logger.exception(
                "Unexpected error extracting tool call. raw=%s", model_output[:300]
            )
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
        if self._request_skips_reasoning(request):
            self.is_reasoning_ended = True
        elif self._request_forces_reasoning(request) and previous_text == "":
            # In vLLM's auto-tool + reasoning streaming path this parser is
            # called only after the reasoning parser has found the assistant
            # delimiter. The delimiter has already been stripped from
            # current_text, so carry that handoff state explicitly.
            self.is_reasoning_ended = True

        # Detect bare array mode: the reasoning parser strips the
        # " -> tool/function_call\n" header before handing off, so current_text
        # begins with "[{" directly. Latch the flag on first detection.
        stripped_text = current_text.lstrip("\n")
        if (
            not self.is_bare_tool_call_mode
            and self.tool_call_start_token not in current_text
            and bool(request.tools)
        ):
            if stripped_text == "[":
                self.pending_bare_tool_call_prefix = stripped_text
                return None
            if stripped_text.startswith("[{"):
                self.bare_tool_call_was_reasoning_ended = self.is_reasoning_ended
                self.is_bare_tool_call_mode = True
                self.is_reasoning_ended = True
                self.pending_bare_tool_call_prefix = ""
            elif self.pending_bare_tool_call_prefix:
                delta_text = self.pending_bare_tool_call_prefix + delta_text
                self.pending_bare_tool_call_prefix = ""

        if self.tool_call_start_token in current_text or self.is_bare_tool_call_mode:
            # flush any buffered reasoning text before switching
            # to tool call parsing, so content accumulated in buffer_string is
            # not silently lost on branch transition
            pending = self._flush_buffered_delta()

            if self.tool_call_start_token in current_text:
                function_call_text = current_text.split(self.tool_call_start_token)[-1]
            else:
                # bare array mode: reasoning parser already stripped the header
                function_call_text = stripped_text
            function_call_text = function_call_text[self.tool_call_offset :]

            # find first { outside strings
            opening_brace_index = None
            for idx, c in enumerate(function_call_text):
                if c == "{":
                    opening_brace_index = idx
                    break

            if opening_brace_index is None:
                return pending

            # use depth+string-aware scanner instead of naive brace search
            closing_brace_index = self._find_json_object_end(
                function_call_text, opening_brace_index
            )

            if closing_brace_index is None:
                return pending

            try:
                raw_function_call = json.loads(
                    function_call_text[opening_brace_index : closing_brace_index + 1]
                )
                if not self._is_function_call_like(raw_function_call, request):
                    if self.is_bare_tool_call_mode:
                        self.is_bare_tool_call_mode = False
                        self.is_reasoning_ended = (
                            self.bare_tool_call_was_reasoning_ended
                        )
                        self.tool_call_offset = 0
                        return (
                            DeltaMessage(content=stripped_text)
                            if self.is_reasoning_ended
                            else self._delta(reasoning=stripped_text)
                        )
                    return pending

                _function_call = self._normalize_function_call(raw_function_call)
                self.current_tool_id += 1
                # accumulate absolute offset, not relative closing index
                self.tool_call_offset = self.tool_call_offset + closing_brace_index + 1
                self.prev_tool_call_arr.append(_function_call)
                arguments = self._arguments_json(_function_call)
                self.streamed_args_for_tool.append(arguments)

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=f"hcx_tool_call_{self.current_tool_id}",
                            function=DeltaFunctionCall(
                                name=_function_call.get("name", ""), arguments=arguments
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )

            except json.JSONDecodeError:
                logger.debug(
                    "Decode error: %s",
                    function_call_text[opening_brace_index : closing_brace_index + 1],
                )
                return pending

        else:
            # removed `len(current_token_ids) == 2` heuristic
            # reasoning termination based on explicit template markers
            if "<|im_end|>\n<|im_start|>assistant" in current_text:
                self.is_reasoning_ended = True

            # set up buffer for special string processing
            self.buffer_string += delta_text
            buffered_content = ""

            if not self.is_reasoning_ended and (
                self._is_partial_tool_role_fragment(current_text)
                or self._is_tool_role_prefix_or_fragment(self.buffer_string)
            ):
                return None

            if (
                not self.is_reasoning_ended
                and not self.has_emitted_reasoning_text
                and self.buffer_string.strip() == ""
            ):
                return None

            if self.check_is_special_string():
                buffered_content, delta_text = self.remove_special_string()
                self.buffer_string = delta_text

                if self.is_reasoning_ended:
                    return self._emit_text_delta(buffered_content)
                else:
                    if self._is_tool_role_prefix_or_fragment(buffered_content):
                        return None
                    if buffered_content and buffered_content.strip():
                        self.has_emitted_reasoning_text = True
                    return self._emit_text_delta(buffered_content)

            if self.check_is_part_of_special_string():
                return None
            else:
                delta_text = self.buffer_string
                self.buffer_string = ""

            if self.is_reasoning_ended:
                return self._emit_text_delta(delta_text)
            if self._is_tool_role_prefix_or_fragment(delta_text):
                return None
            if delta_text and delta_text.strip():
                self.has_emitted_reasoning_text = True
            return self._emit_text_delta(delta_text)

    def _find_first_special_string(self) -> tuple[int, str] | None:
        first_match = None
        for special_string in self.special_strings:
            index = self.buffer_string.find(special_string)
            if index < 0:
                continue
            if first_match is None or index < first_match[0]:
                first_match = (index, special_string)
        return first_match

    def remove_special_string(self) -> tuple[str, str]:
        first_match = self._find_first_special_string()
        if first_match is None:
            return self.buffer_string, ""
        start, special_string = first_match
        end = start + len(special_string)
        return self.buffer_string[:start], self.buffer_string[end:]

    def check_is_special_string(self) -> bool:
        return self._find_first_special_string() is not None

    def check_is_part_of_special_string(self) -> bool:
        for special_string in self.special_strings:
            min_len = min(len(self.buffer_string), len(special_string))
            for length in range(min_len, 0, -1):
                if self.buffer_string[-length:] == special_string[:length]:
                    return True
        return False
