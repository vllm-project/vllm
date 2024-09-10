import json
from json import JSONDecoder
import re
from typing import Dict, List, Sequence, Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall,
                                              InitialDeltaToolCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser)
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


class LlamaToolParser(ToolParser):
    """
    Tool call parser for Llama 3.1 models intended for use with the
    examples/tool_chat_template_llama.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser mistral are all set
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.current_tool_initial_sent: bool = False
        self.streamed_args_for_tool: List[str] = [
        ]  # map what has been streamed for each tool so far to a list
        self.bot_token = "<|python_tag|>"
        self.bot_token_id = self.model_tokenizer.vocab[self.bot_token]
        self.tool_call_regex = re.compile(r"\[{.*?}\]", re.DOTALL)

    def extract_tool_calls(self,
                           model_output: str) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response. Requires
        find-and-replacing single quotes with double quotes for JSON parsing,
        make sure your tool call arguments don't ever include quotes!
        """

        try:
            # load the JSON, and then use it to build the Function and
            # Tool Call
            dec = JSONDecoder()
            function_call_arr = []

            # depending on the prompt format the Llama model may or may not
            # prefix the output with the <|python_tag|> token
            start_idx = len(self.bot_token) if model_output.startswith(self.bot_token) else 0
            while start_idx < len(model_output):
                (obj, end_idx) = dec.raw_decode(model_output[start_idx:])
                start_idx += end_idx + len('; ')
                function_call_arr.append(obj)

            tool_calls: List[ToolCall] = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(raw_function_call["arguments"] \
                                if "arguments" in raw_function_call \
                                else raw_function_call["parameters"])))
                for raw_function_call in function_call_arr
            ]

            # get any content before  the tool call
            ret = ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=None)
            return ret

        except Exception as e:
            logger.error("Error in extracting tool call from response: %s", e)
            print("ERROR", e)
            # return information to just treat the tool call as regular JSON
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
    ) -> Union[DeltaMessage, None]:
        raise NotImplementedError("streaming tool calls not supported for Llama 3.1 yet")
