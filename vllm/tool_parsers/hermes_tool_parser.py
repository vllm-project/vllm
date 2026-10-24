# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import is_complete_json, partial_tag_overlap
from vllm.utils.mistral import is_mistral_tokenizer

logger = init_logger(__name__)


class Hermes2ProToolParser(ToolParser):
    structural_tag_model = "hermes"
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    tool_call_regex = re.compile(
        r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
    )
    scratch_pad_regex = re.compile(r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL)

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if is_mistral_tokenizer(tokenizer):
            logger.error("Detected Mistral tokenizer when using a Hermes model")
            self.model_tokenizer = tokenizer.tokenizer

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Streaming state: what has been sent to the client.
        self._sent_content_idx: int = 0
        self._current_text: str = ""
        # A `<tool_call>` that cannot be a real call. Everything after it,
        # including later tags, is ordinary content.
        self._suppress_tool_calls: bool = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because the tool_call tokens are
            # marked "special" in some models. Since they are skipped
            # prior to the call to the tool parser, it breaks tool calling.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
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

                # load the JSON, and then use it to build the Function and
                # Tool Call
                raw_function_calls = [
                    json.loads(match[0] if match[0] else match[1])
                    for match in function_call_tuples
                ]
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(
                                function_call["arguments"], ensure_ascii=False
                            ),
                        ),
                    )
                    for function_call in raw_function_calls
                ]

                content = model_output[: model_output.find(self.tool_call_start_token)]
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

    def _region_is_prose(self, current_text: str, start: int, *, final: bool) -> bool:
        """Return whether the `<tool_call>` at `start` cannot be a real call.

        A real call is a JSON object, which may still be streaming. A closed
        region with no tool name, or any other text, is prose.
        """
        json_start = start + len(self.tool_call_start_token)
        end = current_text.find(self.tool_call_end_token, json_start)
        if end != -1:
            body = current_text[json_start:end]
            closed = True
        else:
            body = current_text[json_start:]
            overlap = partial_tag_overlap(body, self.tool_call_end_token)
            if overlap:
                body = body[:-overlap]
            closed = False

        stripped = body.strip()
        if not stripped:
            return closed or final
        if stripped[0] != "{":
            return True
        if self._extract_tool_name(stripped):
            return False
        return closed or final

    def _extract_content(self, current_text: str, *, final: bool = False) -> str | None:
        """Return unsent non-tool-call text, or None.

        Holds back any suffix that could be a partial <tool_call> tag, and
        holds back from a `<tool_call>` that may still be a real call. A
        start token that is only prose is sent as content.
        """
        if self._suppress_tool_calls:
            sendable_idx = len(current_text)
        else:
            start = current_text.find(
                self.tool_call_start_token, self._sent_content_idx
            )
            if start == -1:
                overlap_length = (
                    0
                    if final
                    else partial_tag_overlap(current_text, self.tool_call_start_token)
                )
                sendable_idx = len(current_text) - overlap_length
            elif self._region_is_prose(current_text, start, final=final):
                self._suppress_tool_calls = True
                sendable_idx = len(current_text)
            else:
                sendable_idx = start

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_call_jsons(self, text: str) -> list[tuple[str, bool]]:
        """Extract (json_text, is_complete) for each <tool_call> region."""
        results: list[tuple[str, bool]] = []
        pos = 0
        while True:
            start = text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            json_start = start + len(self.tool_call_start_token)
            json_end = text.find(self.tool_call_end_token, json_start)
            if json_end != -1:
                results.append((text[json_start:json_end].strip(), True))
                pos = json_end + len(self.tool_call_end_token)
            else:
                raw = text[json_start:]
                # Strip partial </tool_call> suffix if present.
                overlap = partial_tag_overlap(raw, self.tool_call_end_token)
                if overlap:
                    raw = raw[:-overlap]
                tc_json = raw.strip()
                # Valid JSON without closing tag = complete body,
                # tag tokens just haven't arrived yet.
                is_complete = is_complete_json(tc_json) if tc_json else False
                results.append((tc_json, is_complete))
                break
        return results

    @staticmethod
    def _extract_tool_name(tc_json: str) -> str | None:
        """Extract tool name, or None if the name isn't complete yet."""
        match = re.search(r'"name"\s*:\s*"([^"]+)"', tc_json)
        return match.group(1) if match else None

    @staticmethod
    def _extract_tool_args(tc_json: str, is_complete: bool) -> str | None:
        """Extract tool arguments from the tool call JSON.

        Given {"name": "f", "arguments": {"x": 1}}, returns '{"x": 1}'.
        When is_complete, strips the trailing '}' that closes the outer
        object (not the arguments). For partial JSON, returns as-is.
        """
        match = re.search(r'"arguments"\s*:\s*', tc_json)
        if not match:
            return None
        raw = tc_json[match.end() :]
        if is_complete:
            raw = raw.rstrip()
            if raw.endswith("}"):
                raw = raw[:-1].rstrip()
        return raw

    def _compute_args_diff(
        self, index: int, tc_json: str, is_complete: bool
    ) -> str | None:
        """Return new argument text not yet sent for tool `index`, or None."""
        args = self._extract_tool_args(tc_json, is_complete)
        if args is None or len(args) <= len(self.streamed_args_for_tool[index]):
            return None
        diff = args[len(self.streamed_args_for_tool[index]) :]
        self.streamed_args_for_tool[index] = args
        self.prev_tool_call_arr[index]["arguments"] = args
        return diff

    def finish_streaming(self) -> DeltaMessage | None:
        """Return held text that never became a tool call, at end of stream."""
        if any("name" in call for call in self.prev_tool_call_arr):
            return None
        content = self._extract_content(self._current_text, final=True)
        return DeltaMessage(content=content) if content else None

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
        """Incrementally stream tool call deltas from accumulated output.

        On each invocation, re-parses the full ``current_text`` to find
        ``<tool_call>`` regions, then diffs against previously sent state
        to emit only new content, tool names, or argument fragments.

        Returns a ``DeltaMessage`` containing either plain content (for
        text preceding any tool call) or one or more ``DeltaToolCall``
        entries, or ``None`` if there is nothing new to send yet."""
        try:
            self._current_text = current_text
            # Extract any content before tool calls. A stray `<tool_call>`
            # in prose is content, not a call that never closes.
            content = self._extract_content(current_text)
            if self._suppress_tool_calls:
                return DeltaMessage(content=content) if content else None

            tool_call_jsons = self._extract_tool_call_jsons(current_text)
            tool_call_deltas: list[DeltaToolCall] = []

            for i, (tc_json, is_complete) in enumerate(tool_call_jsons):
                if i >= len(self.prev_tool_call_arr):
                    self.prev_tool_call_arr.append({})
                    self.streamed_args_for_tool.append("")

                # Stream back tool name.
                if "name" not in self.prev_tool_call_arr[i]:
                    name = self._extract_tool_name(tc_json)
                    if not name:
                        # Can't skip to tool i+1 if i isn't ready
                        break
                    self.prev_tool_call_arr[i]["name"] = name
                    tool_call_deltas.append(
                        DeltaToolCall(
                            index=i,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(name=name).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

                # Stream back new tool args by diffing against what was sent.
                args_diff = self._compute_args_diff(i, tc_json, is_complete)
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
