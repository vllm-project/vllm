# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Sequence
from typing import Any, Optional

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("phi4_mini_json")
class Phi4MiniJsonToolParser(ToolParser):
    """
    Tool call parser for phi-4-mini models intended for use with the
    examples/tool_chat_template_llama.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser phi4_mini_json  
    are all set
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: list[dict[str, Any]] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = [
        ]  # map what has been streamed for each tool so far to a list
        self.bot_token: str = "functools"

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        print(f"Model output: {model_output}")

        pattern = r'functools\[(.*?)\]'
        matches = re.search(pattern, model_output, re.DOTALL)

        if not matches:
            print("No function calls found")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            function_call_arr: list[dict[str, Any]] = []
            try:
                json_content = '[' + matches.group(1) + ']'

                function_call_arr = json.loads(json_content)
                print(f"Successfully extracted {len(function_call_arr)} "
                      "function calls")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")

            tool_calls: list[ToolCall] = [
                ToolCall(
                    id=f"chatcmpl-tool-{random_uuid()}",
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(
                            raw_function_call["arguments"] if "arguments" in
                            raw_function_call else
                            raw_function_call["parameters"])))
                for raw_function_call in function_call_arr
            ]

            # get any content before the tool call
            ret = ExtractedToolCallInformation(tools_called=True,
                                               tool_calls=tool_calls,
                                               content=None)
            return ret

        except Exception:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Optional[DeltaMessage]:

        return None
