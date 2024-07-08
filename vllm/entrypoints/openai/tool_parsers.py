from vllm.entrypoints.openai.protocol import ToolCall, FunctionCall, ChatCompletionResponse, ExtractedToolCallInformation
from vllm.logger import init_logger
from typing import List
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
import json
from pydantic import BaseModel
import re
from vllm.entrypoints.openai.protocol import DeltaMessage
logger = init_logger(__name__)


class ToolParser:

    def __init__(self):
        pass

    @staticmethod
    def extract_tool_calls(model_output: str) -> ExtractedToolCallInformation:
        raise NotImplementedError('AbstractToolParser.extract_tool_calls has not been implemented!')

    def extract_tool_calls_streaming(self,
                                     previous_text: str,
                                     current_text: str,
                                     delta_text: str,
                                     previous_token_ids: List[int],
                                     current_token_ids: List[int],
                                     delta_token_ids: List[int],
                                     ) -> DeltaMessage | None:
        raise NotImplementedError('AbstractToolParser.extract_tool_calls_streaming has not been implemented!')


class MistralToolParser(ToolParser):
    bot_token: str = '[TOOL_CALLS]'
    bot_token_id: int = 5

    @staticmethod
    def extract_tool_calls(model_output: str) -> ExtractedToolCallInformation:

        # Get the tool call token from the tokenizer
        if MistralToolParser.bot_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        else:
            try:
                # extract the token so we hopefully have a JSON string
                raw_tool_call = (model_output
                                 .replace(MistralToolParser.bot_token, '') # remove BOT token
                                 .replace("'", '"')) # ... hack to parse broken mistral JSON
                # load the JSON, and then use it to build the Function and Tool Call
                function_call_arr = json.loads(raw_tool_call)
                tool_calls: List[ToolCall] = [
                    ToolCall(
                        type='function',
                        function=FunctionCall(
                            name=raw_function_call['name'],
                            # function call args are JSON but as a string
                            arguments=json.dumps(raw_function_call['arguments'])
                        )
                    )
                    for raw_function_call in function_call_arr
                ]
                content = model_output.split(MistralToolParser.bot_token)[0]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if len(content) > 0 else None
                )

            except Exception as e:
                # TODO discussion on how to best handle invalidly-generated tool calls
                logger.error("Error in extracting tool call from response: %s", e)
                print('ERROR', e)
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )

    def extract_tool_calls_streaming(self,
                                     previous_text: str,
                                     current_text: str,
                                     delta_text: str,
                                     previous_token_ids: List[int],
                                     current_token_ids: List[int],
                                     delta_token_ids: List[int],
                                     ) -> DeltaMessage | None:


        return DeltaMessage(content=delta_text)


class Hermes2ProToolParser(ToolParser):

    tool_call_start: str = '<tool_call>'
    tool_call_end: str = '</tool_call>'

    # regex to match between <tool_call> and </tool_call> OR between <tool_call> and EOS (happens sometimes :))
    tool_call_regex = re.compile(r'<tool_call>(.*?)</tool_call>|<tool_call>(.*)', re.DOTALL)
    scratch_pad_regex = re.compile(r'<scratch_pad>(.*?)</scratch_pad>', re.DOTALL)

    @staticmethod
    def extract_tool_calls(model_output: str) -> ExtractedToolCallInformation:

        # sanity check; avoid unnecessary processing
        if Hermes2ProToolParser.tool_call_start not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

        else:

            try:
                # there are two possible captures - between tags, or between a tag and end-of-string so the result of findall
                #   is an array of tuples where one is a function call and the other is None
                function_call_tuples = Hermes2ProToolParser.tool_call_regex.findall(model_output)

                # load the JSON, and then use it to build the Function and Tool Call
                raw_function_calls = [json.loads(match[0] if match[0] else match[1]) for match in function_call_tuples]
                tool_calls = [
                    ToolCall(
                        type='function',
                        function=FunctionCall(
                            name=function_call['name'],
                            # function call args are JSON but as a string
                            arguments=json.dumps(function_call['arguments'])
                        )
                    ) for function_call in raw_function_calls
                ]
                content_match = Hermes2ProToolParser.scratch_pad_regex.search(model_output)
                content = content_match.group(1) if content_match else None
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None
                )

            except Exception as e:
                logger.error("Error in extracting tool call from response %s", e)
                # TODO discussion on how to best handle invalidly-generated tool calls
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )

    def extract_tool_calls_streaming(self,
                                     previous_text: str,
                                     current_text: str,
                                     delta_text: str,
                                     previous_token_ids: List[int],
                                     current_token_ids: List[int],
                                     delta_token_ids: List[int]
                                     ) -> DeltaMessage:
        raise NotImplementedError('Hermes2ProToolParser.extract_tool_calls_streaming has not been implemented!')
