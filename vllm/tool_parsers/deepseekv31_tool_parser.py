# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

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
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)


class DeepSeekV31ToolParser(ToolParser):
    structural_tag_model = "deepseek_v3_1"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        self.tool_calls_start_token: str = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end_token: str = "<｜tool▁calls▁end｜>"

        self.tool_call_start_token: str = "<｜tool▁call▁begin｜>"
        self.tool_call_end_token: str = "<｜tool▁call▁end｜>"

        self.tool_call_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<function_name>.*?)<｜tool▁sep｜>(?P<function_arguments>.*?)<｜tool▁call▁end｜>"
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<function_name>.*)<｜tool▁sep｜>(?P<function_arguments>.*)"
        )

        self.stream_tool_call_name_regex = re.compile(
            r"(?P<function_name>.*)<｜tool▁sep｜>"
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "DeepSeek-V3.1 Tool parser could not locate tool call "
                "start/end tokens in the tokenizer!"
            )

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

                tool_calls = []
                for match in function_call_tuples:
                    function_name, function_args = match
                    tool_calls.append(
                        ToolCall(
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

    def _parse_stream_tool_calls(self, text: str) -> list[dict]:
        """Parse every tool call in *text*, in order.

        Returns one dict per ``tool_call_start_token`` found: complete
        blocks and the trailing in-progress block alike. ``name`` is
        ``None`` while the model has not finished emitting it.
        """
        calls: list[dict] = []
        pos = 0
        while True:
            start = text.find(self.tool_call_start_token, pos)
            if start == -1:
                break
            body_start = start + len(self.tool_call_start_token)
            end = text.find(self.tool_call_end_token, body_start)
            body = text[body_start:end] if end != -1 else text[body_start:]
            portion = body.rstrip()
            match = self.stream_tool_call_portion_regex.match(portion)
            if match:
                name = match.group("function_name")
                arguments = match.group("function_arguments")
            else:
                name_match = self.stream_tool_call_name_regex.match(portion)
                name = name_match.group("function_name") if name_match else None
                arguments = ""
            calls.append({"name": name, "arguments": arguments})
            if end == -1:
                break
            pos = end + len(self.tool_call_end_token)
        return calls

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
        if self.tool_calls_start_token_id not in current_token_ids:
            return DeltaMessage(content=delta_text)

        delta_text = delta_text.replace(self.tool_calls_start_token, "").replace(
            self.tool_calls_end_token, ""
        )
        try:
            prev_tool_end_count = previous_token_ids.count(self.tool_call_end_token_id)
            cur_tool_start_count = current_token_ids.count(
                self.tool_call_start_token_id
            )
            cur_tool_end_count = current_token_ids.count(self.tool_call_end_token_id)

            # plain text: outside any tool call (before the first one starts,
            # or after the last one has closed)
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                return DeltaMessage(content=delta_text)

            # Re-parse every tool call from the accumulated text and stream
            # whatever has not been sent yet. Iterating all calls (instead of
            # advancing one state per invocation) keeps parsing correct when a
            # single delta carries one or more complete calls, e.g. from
            # speculative decoding or the reasoning-end re-delivery of
            # accumulated text.
            delta_tool_calls: list[DeltaToolCall] = []
            for index, call in enumerate(self._parse_stream_tool_calls(current_text)):
                if index < self.current_tool_id:
                    continue
                if index > self.current_tool_id:
                    self.current_tool_id = index
                    self.current_tool_name_sent = False
                while len(self.streamed_args_for_tool) <= index:
                    self.streamed_args_for_tool.append("")
                while len(self.prev_tool_call_arr) <= index:
                    self.prev_tool_call_arr.append({})

                name = call["name"]
                if name is None:
                    # tool name is still being generated; nothing further can
                    # be streamed for this or later calls
                    break
                arguments = call["arguments"]

                streamed = self.streamed_args_for_tool[index]
                argument_diff = None
                if (
                    arguments
                    and len(arguments) > len(streamed)
                    and arguments.startswith(streamed)
                ):
                    argument_diff = arguments[len(streamed) :]
                    self.streamed_args_for_tool[index] = arguments
                self.prev_tool_call_arr[index] = {
                    "name": name,
                    "arguments": arguments,
                }

                if not self.current_tool_name_sent:
                    self.current_tool_name_sent = True
                    delta_tool_calls.append(
                        DeltaToolCall(
                            index=index,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=name, arguments=argument_diff
                            ).model_dump(exclude_none=True),
                        )
                    )
                elif argument_diff is not None:
                    delta_tool_calls.append(
                        DeltaToolCall(
                            index=index,
                            function=DeltaFunctionCall(
                                arguments=argument_diff
                            ).model_dump(exclude_none=True),
                        )
                    )

            if delta_tool_calls:
                return DeltaMessage(tool_calls=delta_tool_calls)
            return None

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.
