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
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.tool_parsers.utils import safe_json_loads

logger = init_logger(__name__)


class Ernie45ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        """
        Ernie thinking model format:
        abc\n</think>\n\n\n<tool_call>\ndef\n</tool_call>\n
        """
        super().__init__(tokenizer)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []
        self.think_end_token = "</think>"
        self.response_start_token: str = "<response>"
        self.response_end_token: str = "</response>"
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_calls_start_token = self.tool_call_start_token
        self.newline_token: str = "<0x0A>"

        self.tool_call_regex = re.compile(
            r"<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.response_start_token_id = self.vocab.get(self.response_start_token)
        self.response_end_token_id = self.vocab.get(self.response_end_token)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self.newline_token_id = self.vocab.get(self.newline_token)
        self.parser_token_ids = [
            self.think_end_token_id,
            self.response_start_token_id,
            self.response_end_token_id,
        ]

        self._buffer = ""

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
                tool_call_json_list = self.tool_call_regex.findall(model_output)

                tool_calls = []
                for tool_call_json in tool_call_json_list:
                    tool_call_dict = json.loads(tool_call_json)
                    args_str = json.dumps(
                        tool_call_dict.get("arguments", {}), ensure_ascii=False
                    )
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=tool_call_dict.get("name", ""),
                                arguments=args_str,
                            ),
                        )
                    )

                content = model_output[
                    : model_output.find(self.tool_calls_start_token)
                ].rstrip("\n")
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
        self._buffer += delta_text
        cur_text = self._buffer
        start_idx = cur_text.find(self.tool_call_start_token)
        if start_idx == -1:
            self._buffer = ""
            # At least one toolcall has been completed
            if self.current_tool_id > 0:
                cur_text = ""
            if self.current_tool_id == -1 and all(
                token_id == self.newline_token_id for token_id in previous_token_ids
            ):
                cur_text = cur_text.strip("\n")

            # handle <response> </response> when tool_call is not triggered
            # cur_text === delta_text
            content = cur_text
            if self.response_start_token_id in delta_token_ids:
                content = content.lstrip("\n")
                response_start_idx = content.find(self.response_start_token)
                content = content[response_start_idx + len(self.response_start_token) :]
                # if have </response>, remove it
                response_end_idx = content.rfind(self.response_end_token)
                if response_end_idx != -1:
                    content = content[:response_end_idx]
            elif self.response_end_token_id in delta_token_ids:
                response_end_idx = content.rfind(self.response_end_token)
                content = content[:response_end_idx]
            # remove \n after </think> or <response> or </response>
            if (
                len(previous_token_ids) > 0
                and previous_token_ids[-1] in self.parser_token_ids
            ) and (
                len(delta_token_ids) > 0 and delta_token_ids[0] == self.newline_token_id
            ):
                content = content.lstrip("\n")

            return DeltaMessage(content=content if content else None)
        logger.debug("cur_text = %s", cur_text)
        end_idx = cur_text.find(self.tool_call_end_token)
        if end_idx != -1:
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            extracted_tool_calls = self.extract_tool_calls(
                cur_text[: end_idx + len(self.tool_call_end_token)], request
            )

            if len(extracted_tool_calls.tool_calls) == 0:
                logger.warning("Failed to extract any tool calls.")
                return None
            tool_call = extracted_tool_calls.tool_calls[0]
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": tool_call.function.name,
                "arguments": safe_json_loads(tool_call.function.arguments),
            }
            self.streamed_args_for_tool[self.current_tool_id] = (
                tool_call.function.arguments
            )
            delta = DeltaMessage(
                content=extracted_tool_calls.content,
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=DeltaFunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                ],
            )
            self.current_tool_id += 1
            self._buffer = cur_text[end_idx + len(self.tool_call_end_token) :]
            return delta

        self._buffer = cur_text[start_idx:]
        content = cur_text[:start_idx].rstrip("\n")
        return DeltaMessage(content=content if content else None)
