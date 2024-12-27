import io
import json
import re
from typing import Dict, List, Sequence, Union

import ijson

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


@ToolParserManager.register_module("hermes")
class Hermes2ProToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: List[str] = [
        ]  # map what has been streamed for each tool so far to a list

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
        self.scratch_pad_regex = re.compile(
            r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")
        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        if (self.tool_call_start_token_id is None
                or self.tool_call_end_token_id is None):
            raise RuntimeError(
                "Hermes 2 Pro Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")

        self.reset()

    @ijson.coroutine
    def process_tool_call(self):
        while True:
            tool_call = (yield)
            self.in_json = False
            self.tool_calls.append(tool_call)

    def reset(self):
        self.tool_call_start_token_pos = 0
        self.tool_call_end_token_pos = 0
        self.in_tool_call = False
        self.in_json = False
        self.current_tool_id = 0

        self.tool_calls = []

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        # sanity check; avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        else:

            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = (
                    self.tool_call_regex.findall(model_output))

                # load the JSON, and then use it to build the Function and
                # Tool Call
                raw_function_calls = [
                    json.loads(match[0] if match[0] else match[1])
                    for match in function_call_tuples
                ]
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(function_call["arguments"],
                                                 ensure_ascii=False)))
                    for function_call in raw_function_calls
                ]

                content = model_output[:model_output.
                                       find(self.tool_call_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None)

            except Exception:
                logger.exception(
                    "Error in extracting tool call from response.")
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
    ) -> Union[DeltaMessage, None]:
        delta_content = io.StringIO()
        delta_tool_calls = []

        for i in range(len(delta_text)):
            c = delta_text[i]

            # match tool call start token
            if not self.in_tool_call:
                # outer text
                if c == self.tool_call_start_token[
                        self.tool_call_start_token_pos]:
                    self.tool_call_start_token_pos += 1
                    if self.tool_call_start_token_pos == len(
                            self.tool_call_start_token):
                        self.tool_call_start_token_pos = 0
                        self.in_tool_call = True

                        self.json_parser = ijson.items_coro(
                            self.process_tool_call(), "")
                else:
                    delta_content.write(
                        self.tool_call_start_token[:self.
                                                   tool_call_start_token_pos])
                    self.tool_call_start_token_pos = 0
                    delta_content.write(c)
            elif not self.in_json:
                # in tool call but not in json
                if c == "\n":
                    # skip the new line after <tool_call>
                    # and before </tool_call>
                    ...
                elif c == "{":
                    self.in_json = True
                    self.json_parser.send(c.encode("utf-8"))
                elif c == self.tool_call_end_token[
                        self.tool_call_end_token_pos]:
                    self.tool_call_end_token_pos += 1
                    if self.tool_call_end_token_pos == len(
                            self.tool_call_end_token):
                        self.json_parser.close()
                        self.json_parser = None

                        self.tool_call_end_token_pos = 0
                        self.in_tool_call = False
                else:
                    # garbage
                    logger.debug("Unexpected model output: %s", c)
                    self.tool_call_start_token_pos = 0
            else:
                # in tool call and in json
                try:
                    self.json_parser.send(c.encode("utf-8"))
                except ijson.JSONError as e:
                    logger.error("Failed to parse JSON tool call by model: %s",
                                 str(e))

        for tool_call in self.tool_calls:
            logger.info("Got JSON tool call: %s", str(tool_call))

            function_name = tool_call.get("name")
            if function_name is not None:
                arguments = tool_call.get("arguments")
                if arguments is None:
                    # robust response arguments to work around certain
                    # client bugs in case the client doesn't check the
                    # arguments properly
                    arguments = {}
                delta_tool_call = DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                    id=f"chatcmpl-tool-{random_uuid()}",
                    function=DeltaFunctionCall(
                        name=function_name,
                        arguments=json.dumps(arguments)).model_dump(
                            exclude_none=True))
                delta_tool_calls.append(delta_tool_call)
                self.current_tool_id += 1
            else:
                logger.error("'name' field missing from tool call by model")

        self.tool_calls.clear()

        is_eos = delta_token_ids[-1] == self.model_tokenizer.eos_token_id
        if is_eos:
            self.reset()

        delta_content_str = delta_content.getvalue()

        if len(delta_content_str) > 0 or is_eos:
            if len(delta_tool_calls) == 0:
                return DeltaMessage(content=delta_content_str)
            else:
                return DeltaMessage(content=delta_content_str,
                                    tool_calls=delta_tool_calls)
        elif len(delta_tool_calls) > 0:
            return DeltaMessage(tool_calls=delta_tool_calls)
        else:
            return None
