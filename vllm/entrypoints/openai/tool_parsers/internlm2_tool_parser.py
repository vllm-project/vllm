import json
from typing import Sequence, Union

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module(["internlm2", "internlm2_5"])
class Internlm2ToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.position = 0
        self.current_tool_id = 0

    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            request.skip_special_tokens = False
        return request

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
        if '<|action_start|>' not in current_text:
            self.position = len(current_text)
            return DeltaMessage(content=delta_text)

        if self.current_tool_id > 0:
            return DeltaMessage(content='')

        last_pos = self.position
        if '<|action_start|><|plugin|>' not in current_text[last_pos:]:
            return None

        new_delta = current_text[last_pos:]
        text, action = new_delta.split('<|action_start|><|plugin|>')
        if '<|action_end|>' not in action:
            self.position = last_pos + len(text)
            return None if len(text) == 0 else DeltaMessage(content=text)

        action = action.split('<|action_end|>'.strip())[0]

        action = action[action.find('{'):]
        action_dict = json.loads(action)
        name, parameters = action_dict['name'], json.dumps(
            action_dict.get('parameters', action_dict.get('arguments', {})))

        last_pos = current_text[last_pos:].find("<|action_end|>") + len(
            '<|action_end|>')
        self.position = last_pos
        if not request.tools or name not in [
                t.function.name for t in request.tools
        ]:
            return None if len(text) == 0 else DeltaMessage(content=text)

        delta = DeltaMessage(content=text,
                             tool_calls=[
                                 DeltaToolCall(
                                     index=self.current_tool_id,
                                     id=f"chatcmpl-tool-{random_uuid()}",
                                     function=DeltaFunctionCall(
                                         name=name, arguments=parameters)),
                             ])
        self.current_tool_id = self.current_tool_id + 1
        self.prev_tool_call_arr = [action_dict]
        return delta

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        text = model_output
        tools = request.tools
        if '<|action_start|><|plugin|>' in text:
            text, action = text.split('<|action_start|><|plugin|>')
            action = action.split('<|action_end|>'.strip())[0]
            action = action[action.find('{'):]
            action_dict = json.loads(action)
            name, parameters = action_dict['name'], json.dumps(
                action_dict.get('parameters', action_dict.get('arguments',
                                                              {})))

            if not tools or name not in [t.function.name for t in tools]:
                ExtractedToolCallInformation(tools_called=False,
                                             tool_calls=[],
                                             content=text)

            tool_calls = [
                ToolCall(
                    function=FunctionCall(name=name, arguments=parameters))
            ]
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=text if len(text) > 0 else None)

        return ExtractedToolCallInformation(tools_called=False,
                                            tool_calls=[],
                                            content=text)
