# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
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

logger = init_logger(__name__)


def dump_args(args: None | dict[str, Any] | str) -> str | None:
    if args is None or isinstance(args, str):
        return args
    else:
        return json.dumps(args, ensure_ascii=False)


class _FunctionCallCtor(Protocol):
    def __init__(self, *, name: str, arguments: str | None): ...


FuncT = TypeVar("FuncT", bound=_FunctionCallCtor)


class Granite4ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool = list[str]()

        self.look_ahead = ""
        self.in_tc = False

        self.tc_start = "<tool_call>"
        self.tc_end = "</tool_call>"
        self.start_regex = re.compile(self.tc_start)
        self.end_regex = re.compile(self.tc_end)

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

    def _collect_results(
        self, text_segments: list[str], tc_segments: list[str], cls: type[FuncT]
    ) -> tuple[str, list[FuncT]]:
        tool_calls_json: list[dict[str, Any]] = [
            json.loads(tc_text) for tc_text in tc_segments
        ]
        tool_calls = []
        for tc in tool_calls_json:
            assert isinstance(tc, dict)
            self.prev_tool_call_arr.append(tc)
            tool_calls.append(
                cls(
                    name=tc["name"],
                    arguments=dump_args(tc["arguments"]),
                )
            )
        return "".join(text_segments), tool_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        msg = ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )
        try:
            delimiters = [("TC_START", self.tc_start), ("TC_END", self.tc_end)]
            pattern = "|".join(f"(?P<{name}>{pattern})" for name, pattern in delimiters)
            regex = re.compile(pattern)

            text_segments = list[str]()
            tc_segments = list[str]()
            last_cut_loc = 0

            for match in regex.finditer(model_output):
                match_type = match.lastgroup
                if match_type == "TC_START":
                    assert not self.in_tc, "Two tool call start tokens found in a row"
                    if preceding_text := model_output[last_cut_loc : match.start()]:
                        text_segments.append(preceding_text)
                    self.in_tc = True
                elif match_type == "TC_END":
                    assert self.in_tc, (
                        "Tool call end token found without corresponding start token"
                    )
                    tool_text = model_output[last_cut_loc : match.start()]
                    assert tool_text, (
                        "Expected the model to generate text between tool call tokens"
                    )
                    tc_segments.append(tool_text)
                    self.in_tc = False
                else:
                    raise ValueError("Unexpected match")
                last_cut_loc = match.end()
            assert not self.in_tc, "The model generated an incomplete tool call"
            if final_text := model_output[last_cut_loc:]:
                text_segments.append(final_text)

            content, tool_call_funcs = self._collect_results(
                text_segments, tc_segments, FunctionCall
            )
            tool_calls = [
                ToolCall(
                    type="function",
                    function=func,
                )
                for func in tool_call_funcs
            ]
            msg.tools_called = bool(tool_calls)
            msg.tool_calls = tool_calls
            msg.content = content or None
        except Exception:
            logger.exception("Error in extracting tool call from response.")
        return msg

    def _tool_extraction_step(
        self,
        delta_text: str,
    ) -> tuple[bool, str, str]:
        start_token_pos = start_token_end = end_token_pos = end_token_end = -1

        if start_match := self.start_regex.search(delta_text, partial=True):
            if not start_match.partial:
                start_token_pos, start_token_end = start_match.span()
            elif start_match.end() > start_match.start():
                start_token_pos = -2

        if end_match := self.end_regex.search(delta_text):
            end_token_pos, end_token_end = end_match.span()

        # Done means that we've exhausted the current buffer
        # and need more output from the model
        done = True
        content = tc_text = ""

        if start_token_pos < 0:
            # just streaming text so far
            if start_token_pos == -2:
                # There is a partial match
                content = delta_text[: start_match.start()]
                self.look_ahead = delta_text[start_match.start() :]
            else:
                content = delta_text

        elif not self.in_tc:
            # we're entering a new tool call
            self.in_tc = True

            content = delta_text[:start_token_pos]
            if end_token_pos > 0:
                self.start_in_tc = False
                tc_text = delta_text[start_token_end:end_token_pos]
                self.look_ahead = delta_text[end_token_end:]
                done = False  # There could be more content already buffered
            else:
                self.look_ahead = delta_text[start_token_pos:]

        elif end_token_pos < 0:
            # we're in between the start and the end token
            assert self.in_tc
            self.look_ahead = delta_text
        else:
            # We have found the end
            assert self.in_tc
            tc_text = delta_text[start_token_end:end_token_pos]
            self.in_tc = False
            self.look_ahead = delta_text[end_token_end:]
            done = False  # There could be more content already buffered
        return done, content, tc_text

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
            done = False
            text_segments = list[str]()
            tc_segments = list[str]()

            while not done:
                delta_text = self.look_ahead + delta_text
                self.look_ahead = ""
                done, content, tc_text = self._tool_extraction_step(delta_text)
                if content:
                    text_segments.append(content)
                if tc_text:
                    tc_segments.append(tc_text)
                delta_text = ""

            content, tool_call_funcs = self._collect_results(
                text_segments, tc_segments, DeltaFunctionCall
            )

            delta_tool_calls = list[DeltaToolCall]()
            for function in tool_call_funcs:
                self.current_tool_id += 1
                delta_tool_calls.append(
                    DeltaToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        index=self.current_tool_id,
                        function=function.model_dump(exclude_none=True),
                    )
                )
                self.streamed_args_for_tool.append(function.arguments or "")

            assert self.current_tool_id + 1 == len(self.prev_tool_call_arr)
            assert self.current_tool_id + 1 == len(self.streamed_args_for_tool)

            msg = DeltaMessage(content=content or None, tool_calls=delta_tool_calls)
            if msg.content or msg.tool_calls:
                return msg

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
        return None
