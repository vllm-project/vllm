# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.harmony_utils import parse_output_into_messages
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer import AnyTokenizer
else:
    AnyTokenizer = object

logger = init_logger(__name__)


class OpenAIToolParser(ToolParser):
    def __init__(self, tokenizer: "AnyTokenizer"):
        super().__init__(tokenizer)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        if token_ids is None:
            raise NotImplementedError(
                "OpenAIToolParser requires token IDs and does not support text-based extraction."  # noqa: E501
            )

        parser = parse_output_into_messages(token_ids)
        tool_calls = []
        final_content = None

        def _create_tool_call(function_name: str, arguments: str) -> ToolCall:
            # Sanitize the function name to remove leaked tags (e.g. <|channel|>)
            clean_name = function_name.split("<")[0].strip()
            
            try:
                clean_args = json.dumps(json.loads(arguments))
            except json.JSONDecodeError:
                logger.debug("Partial or invalid JSON tool call detected.")
                clean_args = arguments
            
            return ToolCall(
                type="function",
                function=FunctionCall(
                    name=clean_name,
                    arguments=clean_args,
                ),
            )

        if len(parser.messages) > 0:
            for msg in parser.messages:
                if len(msg.content) < 1:
                    continue
                msg_text = msg.content[0].text
                
                if msg.recipient and msg.recipient.startswith("functions."):
                    if not msg.content_type or "json" in msg.content_type:
                        func_name = msg.recipient.split("functions.")[1]
                        tool_calls.append(_create_tool_call(func_name, msg_text))
                elif msg.channel == "final":
                    final_content = msg_text

        if parser.current_content:
            curr_text = parser.current_content
            curr_channel = parser.current_channel
            curr_recipient = parser.current_recipient

            if (curr_channel == "commentary" 
                and curr_recipient 
                and curr_recipient.startswith("functions.")):
                
                func_name = curr_recipient.split("functions.")[1]
                tool_calls.append(_create_tool_call(func_name, curr_text))
            
            elif curr_channel == "final":
                if final_content:
                    final_content += curr_text
                else:
                    final_content = curr_text

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=final_content,
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
        raise NotImplementedError(
            "Not being used, manual parsing in serving_chat.py"  # noqa: E501
        )