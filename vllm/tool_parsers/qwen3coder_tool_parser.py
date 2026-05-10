# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import uuid
from collections.abc import Sequence
from typing import Any

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
from vllm.envs import VLLM_ENFORCE_STRICT_TOOL_CALLING
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_tool_properties,
)

logger = init_logger(__name__)


class Qwen3CoderToolParser(ToolParser):
    supports_required_and_named: bool = not VLLM_ENFORCE_STRICT_TOOL_CALLING

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Qwen3 XML Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        # Store accumulated parameters for type conversion
        self.accumulated_params = {}
        self.streaming_request = None

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        if not isinstance(param_value, str):
            return param_value
        param_schema = param_config.get(param_name, {})
        param_types = extract_types_from_schema(param_schema)
        return coerce_to_schema_type(param_value, param_types)

    def _parse_xml_function_call(self, function_call_str: str) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.find(">")
        # If there's no ">" character, this is not a valid xml function call
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        param_config = find_tool_properties(self.tools, function_name)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)
            ),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Quick check to avoid unnecessary processing
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = [
                self._parse_xml_function_call(function_call_str)
                for function_call_str in function_calls
            ]
            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]  # .rstrip()
            valid_tool_calls = [tc for tc in tool_calls if tc is not None]
            return ExtractedToolCallInformation(
                tools_called=(len(valid_tool_calls) > 0),
                tool_calls=valid_tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _streaming_state_snapshot(self) -> tuple:
        """Snapshot of streaming state used to detect progress.

        Used by the driver loop in ``extract_tool_calls_streaming`` to
        decide whether another call to ``_extract_tool_calls_streaming_step``
        could make further progress.
        """
        return (
            self.current_tool_index,
            self.is_tool_call_started,
            self.header_sent,
            self.in_function,
            self.in_param,
            self.json_started,
            self.json_closed,
            self.param_count,
        )

    def _merge_streaming_deltas(
        self, deltas: list[DeltaMessage]
    ) -> DeltaMessage:
        """Merge multiple ``DeltaMessage`` emitted within a single chunk.

        When ``extract_tool_calls_streaming`` drains its state machine, the
        per-step function may emit several deltas (header, ``{``, params,
        ``}``) for one or more tool calls. The serving layer expects at
        most one ``DeltaMessage`` per chunk, so we concatenate content and
        coalesce per-tool ``arguments`` here.
        """
        content_parts: list[str] = []
        # Preserve insertion order keyed by tool index.
        merged_calls: dict[int, DeltaToolCall] = {}
        for delta in deltas:
            if delta.content:
                content_parts.append(delta.content)
            if not delta.tool_calls:
                continue
            for tc in delta.tool_calls:
                if tc.index not in merged_calls:
                    merged_calls[tc.index] = DeltaToolCall(
                        index=tc.index,
                        id=tc.id,
                        type=tc.type,
                        function=DeltaFunctionCall(
                            name=(tc.function.name if tc.function else None),
                            arguments=(
                                (tc.function.arguments if tc.function else "")
                                or ""
                            ),
                        ),
                    )
                else:
                    existing = merged_calls[tc.index]
                    # Keep first id/name/type; concat any new arguments.
                    if tc.function and tc.function.arguments:
                        if (
                            existing.function is None
                            or existing.function.arguments is None
                        ):
                            existing.function = DeltaFunctionCall(
                                name=(
                                    existing.function.name
                                    if existing.function
                                    else None
                                ),
                                arguments="",
                            )
                        existing.function.arguments += tc.function.arguments

        content_str = "".join(content_parts) if content_parts else None
        tool_calls = list(merged_calls.values())
        return DeltaMessage(
            content=content_str,
            tool_calls=tool_calls if tool_calls else None,
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
        """Drive the per-step state machine until it reaches a fixed point.

        The step function (``_extract_tool_calls_streaming_step``) emits at
        most one event (function header / ``{`` / one parameter / ``}`` /
        advance-to-next-tool) per invocation. With speculative decoding a
        single engine output chunk can deliver many tokens at once -- e.g.
        the tail of one tool call together with a complete next tool call
        plus EOS. Calling the step once per chunk would drop everything
        past the first emission.

        Mirror ``Qwen3XMLToolParser.parse_single_streaming_chunks``: run
        the step in a loop, draining all state transitions implied by the
        new ``delta_text``, then merge the emitted deltas into a single
        ``DeltaMessage``.
        """
        # Store request for type conversion. Reset state once at the start
        # of a new message; do NOT reset inside the step (the driver loop
        # below calls the step repeatedly within the same chunk).
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        first = self._extract_tool_calls_streaming_step(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

        # When externally invoked with an empty delta_text we preserve the
        # original single-shot behavior (EOS-flush / no-op). The driver
        # loop is only meaningful when there is new text to drain.
        if not delta_text:
            return first

        results: list[DeltaMessage] = []
        if first is not None:
            results.append(first)

        # Drain remaining state transitions for this chunk. Cap iterations
        # to defend against a pathological state machine; in practice we
        # expect at most O(params per tool * tools per chunk) steps.
        for _ in range(256):
            prev_snap = self._streaming_state_snapshot()
            delta = self._extract_tool_calls_streaming_step(
                previous_text,
                current_text,
                "",
                previous_token_ids,
                current_token_ids,
                [],
                request,
                drain=True,
            )
            if delta is not None:
                results.append(delta)
            if self._streaming_state_snapshot() == prev_snap:
                # Fixed point: no further progress possible without new
                # tokens. Includes the defensive case where a delta was
                # emitted without a state change (shouldn't happen, but
                # prevents infinite loops).
                break

        if not results:
            return None
        if len(results) == 1:
            return results[0]
        return self._merge_streaming_deltas(results)

    def _extract_tool_calls_streaming_step(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
        drain: bool = False,
    ) -> DeltaMessage | None:
        """Single state-machine step. Emits at most one event.

        ``drain=True`` is set by the driver loop in
        ``extract_tool_calls_streaming`` when re-invoking the step with no
        new text. In that mode we skip the empty-delta_text early-return
        and the trailing ``DeltaMessage(content=delta_text)`` fallback
        (which would otherwise emit empty content).
        """
        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text and not drain:
            # Check if this is an EOS token after all tool calls are complete
            # Check for tool calls in text even if is_tool_call_started
            # is False (might have been reset after processing all tools)
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated
                # prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token
                    ) - current_text.count(self.tool_call_end_token)
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Update accumulated text
        self.accumulated_text = current_text

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # Check if this tool call has ended
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}

                # Check if there are more tool calls
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    # No more tool calls
                    self.is_tool_call_started = False
                # Continue processing next tool
                return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # Check if tool call is starting
            if (
                self.tool_call_start_token_id in delta_token_ids
                or self.tool_call_start_token in delta_text
            ):
                self.is_tool_call_started = True
                # Return any content before the tool call
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[
                        : delta_text.index(self.tool_call_start_token)
                    ]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                # Check if we're between tool calls - skip whitespace
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    # We just ended a tool call, skip whitespace
                    return None
                if drain:
                    # No new tokens during drain; nothing to emit.
                    return None
                # Normal content, no tool call
                return DeltaMessage(content=delta_text)

        # Check if we're between tool calls (waiting for next one)
        # Count tool calls we've seen vs processed
        tool_starts_count = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= tool_starts_count:
            # We're past all tool calls, shouldn't be here
            return None

        # We're in a tool call, find the current tool call portion
        # Need to find the correct tool call based on current_tool_index
        tool_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            tool_start_positions.append(idx)
            idx += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_start_positions):
            # No more tool calls to process yet
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        # Find where this tool call ends (or current position if not ended yet)
        tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : tool_end_idx + len(self.tool_call_end_token)
            ]

        # Looking for function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    # Found complete function name
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    # Always append — each tool call is a separate
                    # invocation even if the function name is the same
                    # (e.g. two consecutive "read" calls).
                    self.prev_tool_call_arr.append(
                        {
                            "name": self.current_function_name,
                            "arguments": "{}",
                        }
                    )

                    # Initialize streamed args tracking for this tool.
                    # The serving layer reads streamed_args_for_tool to
                    # compute remaining arguments at stream end. Without
                    # this, IndexError occurs when the serving layer
                    # accesses streamed_args_for_tool[index].
                    self.streamed_args_for_tool.append("")

                    # Send header with function info
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    name=self.current_function_name, arguments=""
                                ),
                                type="function",
                            )
                        ]
                    )
            return None

        # We've sent header, now handle function body
        if self.in_function:
            # Always send opening brace first, regardless of whether
            # parameter_prefix is in the current delta. With speculative
            # decoding, a single delta may contain both the opening brace
            # and parameter data; skipping "{" here would desync
            # json_started from what was actually streamed.
            if not self.json_started:
                self.json_started = True
                self.streamed_args_for_tool[self.current_tool_index] += "{"
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"),
                        )
                    ]
                )

            # Find all parameter start positions in current tool_text
            param_starts = []
            search_idx = 0
            while True:
                search_idx = tool_text.find(self.parameter_prefix, search_idx)
                if search_idx == -1:
                    break
                param_starts.append(search_idx)
                search_idx += len(self.parameter_prefix)

            # Process ALL complete params in a loop (spec decode fix).
            # With speculative decoding a single delta can deliver
            # multiple complete parameters at once. The old single-pass
            # code would process one and ``return None`` if the next was
            # incomplete — skipping any already-complete params that
            # preceded it. Using a loop with ``break`` instead ensures
            # we emit every complete parameter before yielding control.
            json_fragments = []
            while not self.in_param and self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" not in remaining:
                    break

                name_end = remaining.find(">")
                current_param_name = remaining[:name_end]

                value_start = param_start + name_end + 1
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                param_end_idx = value_text.find(self.parameter_end_token)
                if param_end_idx == -1:
                    next_param_idx = value_text.find(self.parameter_prefix)
                    func_end_idx = value_text.find(self.function_end_token)

                    if next_param_idx != -1 and (
                        func_end_idx == -1 or next_param_idx < func_end_idx
                    ):
                        param_end_idx = next_param_idx
                    elif func_end_idx != -1:
                        param_end_idx = func_end_idx
                    else:
                        # Fallback for malformed XML where </function>
                        # is missing. Use </tool_call> as a delimiter
                        # if present in the value so we don't include
                        # the closing tag as part of the param value.
                        tool_end_in_value = value_text.find(self.tool_call_end_token)
                        if tool_end_in_value != -1:
                            param_end_idx = tool_end_in_value
                        else:
                            # Parameter incomplete — break so we still
                            # emit any fragments accumulated by earlier
                            # loop iterations.
                            break

                if param_end_idx == -1:
                    break

                param_value = value_text[:param_end_idx]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                self.current_param_name = current_param_name
                self.accumulated_params[current_param_name] = param_value

                param_config = find_tool_properties(
                    self.tools, self.current_function_name or ""
                )

                converted_value = self._convert_param_value(
                    param_value,
                    current_param_name,
                    param_config,
                    self.current_function_name or "",
                )

                serialized_value = json.dumps(converted_value, ensure_ascii=False)

                if self.param_count == 0:
                    json_fragment = f'"{current_param_name}": {serialized_value}'
                else:
                    json_fragment = f', "{current_param_name}": {serialized_value}'

                self.param_count += 1
                json_fragments.append(json_fragment)

            if json_fragments:
                combined = "".join(json_fragments)

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += combined
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments=combined),
                        )
                    ]
                )

            # Check for function end AFTER processing parameters.
            # This ordering is critical: with speculative decoding a
            # burst can deliver the final parameter value together with
            # </function>. If the close check ran first it would emit
            # "}" and set in_function=False before the parameter loop
            # ever ran, causing the parameter to be silently dropped.
            #
            # Symmetric fallback with the parameter-end detection above:
            # if </function> is absent but </tool_call> is present, the
            # function has implicitly ended. This handles:
            #   * Tokenizers where </function> / </parameter> are added
            #     special tokens stripped under skip_special_tokens=True
            #     (so the literal substring never appears in current_text).
            #   * Malformed model output that emits </tool_call> without
            #     a preceding </function>.
            # Without this fallback the parameter is still emitted (the
            # param-end logic above already falls back to </tool_call>),
            # but the closing "}" is silently dropped.
            function_ended = (
                self.function_end_token in tool_text
                or self.tool_call_end_token in tool_text
            )
            if not self.json_closed and function_ended:
                self.json_closed = True

                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = tool_text.find(self.function_end_token, func_start)
                if func_content_end == -1:
                    # </function> stripped/missing; bound the function
                    # content at </tool_call> instead so prev_tool_call_arr
                    # gets the correct serialized arguments.
                    func_content_end = tool_text.find(
                        self.tool_call_end_token, func_start
                    )
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            func_content,
                        )
                        if parsed_tool and self.current_tool_index < len(
                            self.prev_tool_call_arr
                        ):
                            self.prev_tool_call_arr[self.current_tool_index][
                                "arguments"
                            ] = parsed_tool.function.arguments
                    except Exception:
                        logger.debug(
                            "Failed to parse tool call during streaming: %s",
                            tool_text,
                            exc_info=True,
                        )

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += "}"
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )

                result = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="}"),
                        )
                    ]
                )

                self.in_function = False
                self.json_closed = True
                self.accumulated_params = {}

                return result

        return None

    def get_structural_tag(self, request: ChatCompletionRequest):
        return get_model_structural_tag(
            model="qwen_3_5",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )
