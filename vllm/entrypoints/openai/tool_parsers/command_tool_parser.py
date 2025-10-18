# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Union

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

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
    ToolParserManager,
)
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("command")
class CommandToolParser(ToolParser):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        # Streaming state
        self.current_tool_id: int = -1

        # Action delimiters
        self.tool_call_start_token = "<|START_ACTION|>"
        self.tool_call_end_token = "<|END_ACTION|>"
        self.tool_call_regex = re.compile(
            r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.DOTALL
        )

        # Precompute token ids
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "CommandToolParser cannot find start/end tokens in vocab"
            )

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        # Synchronous parsing: look for full action block
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        try:
            match = self.tool_call_regex.search(model_output)
            if not match:
                raise ValueError("No action block found")
            payload = match.group(1)
            raw_calls = json.loads(payload)
            tool_calls = []
            for entry in raw_calls:
                name = entry.get("tool_name")
                params = entry.get("parameters", {})
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name, arguments=json.dumps(params, ensure_ascii=False)
                        ),
                    )
                )
            # content before action
            prefix = model_output.split(self.tool_call_start_token, 1)[0]
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=prefix or None
            )
        except Exception:
            logger.exception("Error extracting sync tool calls")
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
    ) -> Union[DeltaMessage, None]:
        prev_start = previous_token_ids.count(self.tool_call_start_token_id)
        cur_start = current_token_ids.count(self.tool_call_start_token_id)
        cur_end = current_token_ids.count(self.tool_call_end_token_id)

        # Case 1: Block not started → Text as is
        if cur_start == 0:
            return DeltaMessage(content=delta_text)

        # Case 2: Starting a new block
        if cur_start > prev_start:
            self.current_tool_id += 1
            return None

        # Case 3: Inside block, not closed → ignored
        if cur_start > cur_end:
            return None

        # Case 4: Block End Point
        if cur_start == cur_end and self.tool_call_end_token in delta_text:
            full = current_text + delta_text

            payload = (
                full.split(self.tool_call_start_token, 1)[1]
                .split(self.tool_call_end_token, 1)[0]
                .strip()
            )
            try:
                calls = partial_json_parser.loads(payload or "[]", Allow.ALL)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("Waiting for complete JSON")
                return None
            except json.JSONDecodeError:
                logger.debug("Malformed JSON payload: %s", payload)
                return None

            calls_list = calls if isinstance(calls, list) else [calls]
            deltas = []
            for entry in calls_list:
                name = entry.get("tool_name")
                params = entry.get("parameters", {})
                args = json.dumps(params, ensure_ascii=False)
                deltas.append(
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=f"chatcmpl-tool-{random_uuid()}",
                        function=DeltaFunctionCall(
                            name=name,
                            arguments=args,
                        ).model_dump(exclude_none=True),
                    )
                )

                self.current_tool_id += 1

            return DeltaMessage(tool_calls=deltas)

        return DeltaMessage(content=delta_text)
