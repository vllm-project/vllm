from vllm.entrypoints.openai.protocol import ToolCall, FunctionCall, ChatCompletionResponse
from vllm.logger import init_logger
from typing import List
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
import json
from pydantic import BaseModel

logger = init_logger(__name__)


class ToolParser:

    def __init__(self):
        pass

    def extract_tool_calls(self, model_response: ChatCompletionResponse) -> List[ToolCall]:
        """
        Abstract method intended to be used for extracting tool calls for use in a NON-STREAMING response
        """
        raise NotImplementedError('AbstractToolParser.extract_tool_calls has not been implemented!')

    def extract_tool_calls_streaming(self, generator):
        raise NotImplementedError('AbstractToolParser.extract_tool_calls_streaming has not been implemented!')


class MistralToolParser(ToolParser):
    bot_token: str = '[TOOL_CALLS]'

    def extract_tool_calls(self, model_response: ChatCompletionResponse) -> List[ToolCall]:

        # Get the tool call token from the tokenizer
        if self.bot_token not in model_response.choices[0].message.content:
            return []
        else:
            try:
                # extract the token so we hopefully have a JSON string
                raw_tool_call = (model_response.choices[0].message.content
                                 .replace(MistralToolParser.bot_token, '') # remove BOT token
                                 .replace("'", '"')) # ... hack to parse broken mistral JSON
                tool_call_arr = json.loads(raw_tool_call)
                print('tool call array', tool_call_arr)
                function_calls: List[FunctionCall] = [FunctionCall.parse_obj(obj) for obj in tool_call_arr]
                print('got mistral tool calls', function_calls)
                tool_calls = [ToolCall(type='function', function=function_call) for function_call in function_calls]
                return tool_calls

            except Exception as e:
                logger.error("Error in extracting tool call from response: %s", e)
                return []

    def extract_tool_calls_streaming(self, generator):
        raise NotImplementedError('MistralToolParser.extract_tool_calls_streaming has not been implemented!')


class Hermes2ProToolParser(ToolParser):
    def extract_tool_calls_streaming(self, generator):
        raise NotImplementedError('Hermes2ProToolParser.extract_tool_calls_streaming has not been implemented!')

    def extract_tool_calls(self, model_response: ChatCompletionResponse) -> List[ToolCall]:
        raise NotImplementedError('Hermes2ProToolParser.extract_tool_calls has not been implemented!')