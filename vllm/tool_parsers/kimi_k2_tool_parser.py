# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# code modified from deepseekv3_tool_parser.py

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
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class KimiK2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        # Section-level state management to prevent token leakage
        self.in_tool_section: bool = False
        self.token_buffer: str = ""
        # Buffer size: increased to handle large tool call arguments (file paths,
        # nested JSON). 4096 bytes provides headroom for most practical cases.
        self.buffer_max_size: int = 4096
        self.section_char_count: int = 0  # Track characters processed in tool section
        self.max_section_chars: int = 8192  # Force exit if section exceeds this
        self._buffer_overflow_logged: bool = False  # Log overflow once per session

        # Support both singular and plural variants
        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"
        self.tool_calls_start_token_variants: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",  # singular variant
        ]
        self.tool_calls_end_token_variants: list[str] = [
            "<|tool_calls_section_end|>",
            "<|tool_call_section_end|>",  # singular variant
        ]

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

        # All markers that should be stripped from content
        # Includes tool markers and thinking markers that may leak
        self.all_tool_markers: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call_section_begin|>",
            "<|tool_call_section_end|>",
            "<|tool_call_begin|>",
            "<|tool_call_end|>",
            "<|tool_call_argument_begin|>",
        ]

        # Regex pattern to strip thinking blocks: <think>...</think>
        # This handles both complete thinking blocks and partial markers
        self.thinking_pattern = re.compile(
            r"<think>.*?</think>|<think>|</think>",
            re.DOTALL
        )

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<tool_call_id>.+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)"
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<tool_call_id>.+:\d+)\s*")

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        # Get token IDs for all variants
        self.tool_calls_start_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_start_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]
        self.tool_calls_end_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_end_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "Kimi-K2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

    def _check_and_strip_markers(self, text: str) -> tuple[str, bool, bool]:
        """
        Check for section begin/end markers in text and strip them.
        Returns: (cleaned_text, found_section_begin, found_section_end)
        """
        found_begin = False
        found_end = False
        cleaned = text

        # Check for section begin markers (any variant)
        for variant in self.tool_calls_start_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_begin = True

        # Check for section end markers (any variant)
        for variant in self.tool_calls_end_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_end = True
        return cleaned, found_begin, found_end

    def _strip_all_tool_markers(self, text: str) -> str:
        """
        Strip ALL tool-related markers and thinking blocks from text.
        This prevents any marker from leaking into content.
        """
        cleaned = text
        for marker in self.all_tool_markers:
            cleaned = cleaned.replace(marker, "")
        # Also strip thinking blocks that may leak into content
        cleaned = self.thinking_pattern.sub("", cleaned)
        return cleaned

    def _clean_content(self, text: str) -> str | None:
        """
        Clean content before returning to client.
        Strips "(no content)" placeholder and returns None if empty.
        """
        if text is None:
            return None
        # Strip "(no content)" placeholder that model sometimes generates
        cleaned = text.replace("(no content)", "")
        # Return None if content is empty after cleaning
        if not cleaned.strip():
            return None
        return cleaned

    def _reset_section_state(self) -> None:
        """Reset state when exiting tool section."""
        self.in_tool_section = False
        self.token_buffer = ""
        self.section_char_count = 0

    def reset_streaming_state(self) -> None:
        """
        Reset all streaming state. Call this between requests to prevent
        state leakage when parser instance is reused.
        """
        # Reset section state
        self._reset_section_state()

        # Reset parent class state
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []

        logger.debug("Streaming state reset")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            # Filter "(no content)" from non-tool responses
            cleaned_output = self._clean_content(model_output) or ""
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_output
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
                # Filter "(no content)" from tool call responses
                cleaned_content = self._clean_content(content) or ""
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=cleaned_content,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
                # Filter "(no content)" from error fallback
                cleaned_output = self._clean_content(model_output) or ""
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=cleaned_output
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
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)

        # Flag to defer section exit until after tool parsing completes
        deferred_section_exit = False

        # Add delta to buffer for split marker detection
        self.token_buffer += delta_text

        # Enforce buffer size limit to prevent memory issues
        if len(self.token_buffer) > self.buffer_max_size:
            if not self._buffer_overflow_logged:
                logger.warning(
                    "Token buffer exceeded max size (%d bytes), flushing excess. "
                    "This may indicate very long markers or unusual tokenization.",
                    self.buffer_max_size,
                )
                self._buffer_overflow_logged = True
            # Keep only the most recent content that might contain partial markers
            self.token_buffer = self.token_buffer[-self.buffer_max_size // 2 :]

        # Check buffer for section markers (handles split tokens)
        buffered_text, found_section_begin, found_section_end = (
            self._check_and_strip_markers(self.token_buffer)
        )

        # Track section state transitions
        if found_section_begin and not self.in_tool_section:
            logger.debug("Entering tool section")
            self.in_tool_section = True
            self.token_buffer = buffered_text  # Use cleaned buffer
            self.section_char_count = 0  # Reset counter for new section

            # CRITICAL: Extract content that appeared BEFORE the section marker
            # and return it as reasoning content. This prevents content from
            # leaking into the tool call stream.
            content_before_marker = None
            for variant in self.tool_calls_start_token_variants:
                if variant in delta_text:
                    marker_pos = delta_text.find(variant)
                    if marker_pos > 0:
                        content_before_marker = delta_text[:marker_pos]
                    break

            if content_before_marker and content_before_marker.strip():
                cleaned = self._clean_content(content_before_marker)
                if cleaned:
                    logger.debug(
                        "Returning content before tool section: '%s'",
                        cleaned[:50] if len(cleaned) > 50 else cleaned
                    )
                    return DeltaMessage(content=cleaned)
            # If no content before marker, continue to process tool calls
            # but don't return content from this delta
            return None

        if found_section_end and self.in_tool_section:
            logger.debug("Detected section end marker")
            # CRITICAL: Don't exit early if tool_call_end is in this chunk.
            # Tool parser must emit final arguments/close first to avoid dropping
            # the final tool update and leaking tokens into reasoning channel.
            has_tool_end = self.tool_call_end_token_id in delta_token_ids
            if has_tool_end:
                # Defer exit until after tool parsing completes
                deferred_section_exit = True
                logger.debug("Deferring section exit: tool_call_end in same chunk")
                self.token_buffer = buffered_text
            else:
                # No tool call ending, safe to exit immediately
                logger.debug("Exiting tool section")
                self._reset_section_state()
                # Extract any content AFTER the section end marker in delta_text
                # (don't use buffered_text as it contains tool call data)
                post_section_content = ""
                for variant in self.tool_calls_end_token_variants:
                    if variant in delta_text:
                        parts = delta_text.split(variant, 1)
                        if len(parts) > 1:
                            post_section_content = parts[1]
                        break
                cleaned = self._clean_content(post_section_content)
                if cleaned:
                    return DeltaMessage(content=cleaned)
                # Return None to skip empty chunks (avoids "(no content)" in clients)
                return None
        else:
            self.token_buffer = buffered_text

        # Check if any variant of section start token is in current_token_ids
        has_section_token = any(
            tid in current_token_ids for tid in self.tool_calls_start_token_ids
        )

        # Early return: if no section token detected yet, return as reasoning content
        if not has_section_token and not self.in_tool_section:
            logger.debug("No tool call tokens found!")
            # Don't clear buffer - it needs to accumulate partial markers across deltas
            # Buffer overflow is already protected by lines 215-224
            cleaned = self._clean_content(delta_text)
            return DeltaMessage(content=cleaned) if cleaned else None

        # Strip section markers from delta_text for subsequent processing
        # NOTE: This preprocessing happens BEFORE the regex-based tool call
        # parsing (from PR #24847) to ensure markers are removed cleanly
        # before pattern matching. No double-stripping occurs because
        # section markers and tool call markers are distinct.
        delta_text, _, _ = self._check_and_strip_markers(delta_text)

        # Error recovery: If in tool section for too long, force exit
        if self.in_tool_section:
            self.section_char_count += len(delta_text)
            if self.section_char_count > self.max_section_chars:
                logger.warning(
                    "Tool section exceeded max length (%d chars), forcing exit. "
                    "This may indicate malformed model output.",
                    self.max_section_chars,
                )
                self._reset_section_state()
                # Deferred exit already handled by forced exit above
                # Strip all markers before returning as content
                cleaned_text = self._strip_all_tool_markers(delta_text)
                cleaned_text = self._clean_content(cleaned_text)
                # Return remaining content as reasoning, or None for empty content
                return DeltaMessage(content=cleaned_text) if cleaned_text else None

        try:
            # figure out where we are in the parsing by counting tool call
            # start & end tags
            prev_tool_start_count = previous_token_ids.count(
                self.tool_call_start_token_id
            )
            prev_tool_end_count = previous_token_ids.count(self.tool_call_end_token_id)
            cur_tool_start_count = current_token_ids.count(
                self.tool_call_start_token_id
            )
            cur_tool_end_count = current_token_ids.count(self.tool_call_end_token_id)
            tool_call_portion = None
            text_portion = None

            # case: if we're generating text, OR rounding out a tool call
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                # Suppress content between section begin and first tool begin
                # (header noise). Don't suppress content between tools to avoid
                # breaking potential delimiter characters.
                if self.in_tool_section and cur_tool_start_count == 0:
                    logger.debug(
                        "In tool section before first tool, suppressing: %s",
                        delta_text,
                    )
                    # Return None to skip this chunk (avoids "(no content)" in clients)
                    return None
                logger.debug("Generating text content! skipping tool parsing.")
                cleaned = self._clean_content(delta_text)
                return DeltaMessage(content=cleaned) if cleaned else None

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text + delta_text
                tool_call_portion = (
                    full_text.split(self.tool_call_start_token)[-1]
                    .split(self.tool_call_end_token)[0]
                    .rstrip()
                )
                delta_text = delta_text.split(self.tool_call_end_token)[0].rstrip()
                text_portion = delta_text.split(self.tool_call_end_token)[-1].lstrip()

            # case -- we're starting a new tool call
            if (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count
            ):
                if len(delta_token_ids) > 1:
                    tool_call_portion = current_text.split(self.tool_call_start_token)[
                        -1
                    ]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None

                # set cursors and state appropriately
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # case -- we're updating an existing tool call
            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                # get the portion of the text that's the tool call
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                text_portion = None

            # case -- the current tool call is being closed.
            elif (
                cur_tool_start_count == cur_tool_end_count
                and cur_tool_end_count >= prev_tool_end_count
            ):
                if (self.prev_tool_call_arr is None
                    or len(self.prev_tool_call_arr) == 0
                    or self.current_tool_id >= len(self.prev_tool_call_arr)):
                    logger.debug(
                        "attempting to close tool call, but no tool call or invalid index"
                    )
                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff:
                    diff = (
                        diff.encode("utf-8").decode("unicode_escape")
                        if diff is str
                        else diff
                    )
                    if '"}' not in delta_text:
                        # Handle deferred section exit before returning
                        if deferred_section_exit and self.in_tool_section:
                            self._reset_section_state()
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    logger.debug(
                        "Finishing tool and found diff that had not "
                        "been streamed yet: %s",
                        diff,
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        logger.debug("Completing deferred section exit")
                        self._reset_section_state()
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=diff).model_dump(
                                    exclude_none=True
                                ),
                            )
                        ]
                    )

            # case -- otherwise we're just generating text
            else:
                # Check if we're in tool section - if so, suppress
                if self.in_tool_section:
                    logger.debug("In tool section, suppressing text generation")
                    # Handle deferred section exit before returning
                    if deferred_section_exit:
                        self._reset_section_state()
                    # Return None to skip this chunk (avoids "(no content)" in clients)
                    return None
                # Strip ALL tool markers, not just start/end
                text = self._strip_all_tool_markers(delta_text)
                # Clean and skip empty content after stripping
                text = self._clean_content(text)
                if not text:
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None
                delta = DeltaMessage(content=text)
                # Handle deferred section exit before returning
                if deferred_section_exit and self.in_tool_section:
                    self._reset_section_state()
                return delta

            current_tool_call = dict()
            if tool_call_portion:
                # Strip leading/trailing whitespace to fix regex matching
                tool_call_portion = tool_call_portion.strip()
                current_tool_call_matches = self.stream_tool_call_portion_regex.match(
                    tool_call_portion
                )
                if current_tool_call_matches:
                    tool_id, tool_args = current_tool_call_matches.groups()
                    tool_name = tool_id.split(":")[0].split(".")[-1]
                    current_tool_call["id"] = tool_id.strip()
                    current_tool_call["name"] = tool_name
                    current_tool_call["arguments"] = tool_args
                else:
                    current_tool_call_name_matches = (
                        self.stream_tool_call_name_regex.match(tool_call_portion)
                    )
                    if current_tool_call_name_matches:
                        (tool_id_str,) = current_tool_call_name_matches.groups()
                        tool_name = tool_id_str.split(":")[0].split(".")[-1]
                        current_tool_call["id"] = tool_id_str.strip()
                        current_tool_call["name"] = tool_name
                        current_tool_call["arguments"] = ""
                    else:
                        logger.debug("Not enough token")
                        return None

            # case - we haven't sent the tool name yet. If it's available, send
            #   it. otherwise, wait until it's available.
            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name: str | None = current_tool_call.get("name")
                tool_id = current_tool_call.get("id")
                if function_name:
                    self.current_tool_name_sent = True

                    # CRITICAL: Initialize prev_tool_call_arr before returning
                    # This prevents IndexError on subsequent iterations
                    if len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append(current_tool_call)
                    else:
                        self.prev_tool_call_arr[self.current_tool_id] = current_tool_call

                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=tool_id,
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    return None

            # case -- otherwise, send the tool call delta

            # if the tool call portion is None, send the delta as text
            if tool_call_portion is None:
                # if there's text but not tool calls, send that -
                # otherwise None to skip chunk
                # CRITICAL: Never return content if we're in a tool section
                if self.in_tool_section:
                    return None
                if text_portion is not None:
                    cleaned = self._clean_content(delta_text)
                    delta = DeltaMessage(content=cleaned) if cleaned else None
                else:
                    delta = None
                return delta

            # now, the nitty-gritty of tool calls
            # now we have the portion to parse as tool call.

            logger.debug(
                "Trying to parse current tool call with ID %s", self.current_tool_id
            )

            # if we're starting a new tool call, push an empty object in as
            #   a placeholder for the arguments
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # main logic for tool parsing here - compare prev. partially-parsed
            #   JSON to the current partially-parsed JSON
            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            cur_arguments = current_tool_call.get("arguments")

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            # case -- no arguments have been created yet. skip sending a delta.
            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None

            # case -- prev arguments are defined, but non are now.
            #   probably impossible, but not a fatal error - just keep going
            elif not cur_arguments and prev_arguments:
                logger.error(
                    "should be impossible to have arguments reset "
                    "mid-call. skipping streaming anything."
                )
                delta = None

            # case -- we now have the first info about arguments available from
            #   autocompleting the JSON
            elif cur_arguments and not prev_arguments:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=cur_arguments
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] = cur_arguments

            # last case -- we have an update to existing arguments.
            elif cur_arguments and prev_arguments:
                if (
                    isinstance(delta_text, str)
                    and cur_arguments != prev_arguments
                    and len(cur_arguments) > len(prev_arguments)
                    and cur_arguments.startswith(prev_arguments)
                ):
                    delta_arguments = cur_arguments[len(prev_arguments) :]
                    logger.debug("got diff %s", delta_text)

                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=delta_arguments
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = cur_arguments
                else:
                    delta = None

            # handle saving the state for the current tool into
            # the "prev" list for use in diffing for the next iteration
            if self.current_tool_id < len(self.prev_tool_call_arr):
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            # Handle deferred section exit after tool parsing completes
            if deferred_section_exit and self.in_tool_section:
                logger.debug("Completing deferred section exit")
                self._reset_section_state()

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            # Handle deferred section exit before returning
            if deferred_section_exit and self.in_tool_section:
                self._reset_section_state()
            return None  # do not stream a delta. skip this token ID.
