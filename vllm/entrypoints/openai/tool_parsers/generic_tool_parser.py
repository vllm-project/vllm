from typing import Union

import json

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import (
    MistralToolCall)
from pydantic import TypeAdapter

logger = init_logger(__name__)


@ToolParserManager.register_module("generic")
class GenericToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.tool_call_class = MistralToolCall if isinstance(
            tokenizer, MistralTokenizer) else ToolCall

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        logger.info(f"----------------{model_output}----------------")
        try:
            function_calls = json.loads(model_output)  # Validate JSON format
            tool_calls = []
            content = ""
            for f in function_calls:
                if isinstance(f, dict) and f.get("name"):
                    tool_calls.append(
                        self.tool_call_class(function=FunctionCall(
                            name=f.get("name"),
                            arguments=json.dumps(f.get("arguments", {}),
                                                 ensure_ascii=False),
                        )))
                elif isinstance(f, str):
                    content += f
                else:
                    content += json.dumps(f, ensure_ascii=False)
            return ExtractedToolCallInformation(tools_called=len(tool_calls)
                                                > 0,
                                                tool_calls=tool_calls,
                                                content=content)

        except Exception:
            logger.exception("Error in extracting tool call from response.")
        return ExtractedToolCallInformation(tools_called=False,
                                            tool_calls=[],
                                            content=model_output)

    def extract_tool_calls_streaming(self, previous_text, current_text,
                                     delta_text, previous_token_ids,
                                     current_token_ids, delta_token_ids,
                                     request):
        print(f"delta_text: {delta_text}")
        print(f"previous_text {previous_text} ")
        print(f"current_text {current_text} ")
        # delta = DeltaMessage(tool_calls=[
        #     DeltaToolCall(index=self.current_tool_id,
        #                   function=DeltaFunctionCall(
        #                       arguments=delta_text).model_dump(
        #                           exclude_none=True))
        # ])
        delta = DeltaMessage(content=delta_text)
        return delta
