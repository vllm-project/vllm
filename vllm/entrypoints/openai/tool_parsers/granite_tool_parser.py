import json
import re
from typing import Sequence, Union

from vllm.entrypoints.openai.protocol import (DeltaMessage,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


class GraniteToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.bot_token = "<function_call>"
        self.tool_start_token = self.bot_token
        self.tool_call_regex = re.compile(r"<function_call>\s*")

    def extract_tool_calls(self,
                           model_output: str) -> ExtractedToolCallInformation:
        if self.tool_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        else:
            try:
                matches = list(self.tool_call_regex.finditer(model_output))
                logger.debug("Found %d tool call matches", len(matches))

                raw_function_calls = []

                for i, match in enumerate(matches):
                    # position after the <function_call> tag
                    start_of_json = match.end()
                    # end_index == the start of the next function call
                    # (if exists)
                    next_function_call_start = (matches[i + 1].start() if i +
                                                1 < len(matches) else None)

                    # extract the full JSON object using bracket matching via
                    # start_of_json and optional end index
                    full_json_str = extract_full_json(
                        model_output, start_of_json, next_function_call_start)
                    raw_function_calls.append(json.loads(full_json_str))

                logger.debug("Extracted %d tool calls",
                             len(raw_function_calls))
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(function_call["arguments"]),
                        ),
                    ) for function_call in raw_function_calls
                ]

                content = model_output[:model_output.find(self.bot_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception as e:
                logger.error("Error in extracting tool call from response %s",
                             e)
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
        raise NotImplementedError(
            "streaming tool call parsing has not yet been implemented"
            "for Granite")


def extract_full_json(text, start_index, end_index=None):
    """
    Extracts the full JSON object from text starting at `start_index` 
    and optionally up to `end_index`.
    """
    brace_count = 0
    json_start = start_index
    for i, char in enumerate(text[start_index:end_index], start=start_index):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
        if brace_count == 0:
            return text[json_start:i + 1]
    raise ValueError("Unbalanced braces in the input string")
