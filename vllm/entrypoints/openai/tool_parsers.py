from vllm.entrypoints.openai.protocol import ToolCall, FunctionCall, ChatCompletionResponse, \
    ExtractedToolCallInformation, DeltaToolCall, InitialDeltaToolCall, DeltaFunctionCall
from vllm.logger import init_logger
from typing import List, Dict
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
import json
import partial_json_parser
from partial_json_parser import Allow
import re
from vllm.entrypoints.openai.protocol import DeltaMessage

logger = init_logger(__name__)


def find_common_prefix(s1: str, s2: str) -> str:
    prefix = ''
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def find_common_suffix(s1: str, s2: str) -> str:
    suffix = ''
    min_length = min(len(s1), len(s2))
    for i in range(1, min_length + 1):
        if s1[-i] == s2[-i]:
            suffix = s1[-i] + suffix
        else:
            break
    return suffix


def extract_intermediate_diff(s1: str, s2: str) -> str:
    """
    Extract the difference in the middle between two strings that are KNOWN to have a common prefix and OPTIONALLY
    also a common suffix
    """
    prefix = find_common_prefix(s1, s2)
    suffix = find_common_suffix(s1, s2)
    diff = s1
    if len(prefix):
        diff = diff.replace(prefix, '', 1) # replace the prefix only once in case it's mirrored
    if len(suffix):
        diff = diff[::-1].replace(suffix[::-1], '', 1)[::-1]
    return diff


def find_all_indices(string, substring):
    indices = []
    index = -1
    while True:
        index = string.find(substring, index + 1)
        if index == -1:
            break
        indices.append(index)
    return indices


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
                                 .replace(MistralToolParser.bot_token, '')  # remove BOT token
                                 .replace("'", '"'))  # ... hack to parse broken mistral JSON
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

    def __init__(self):
        super().__init__()
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.current_tool_initial_sent: bool = False

    def extract_tool_calls_streaming(self,
                                     previous_text: str,
                                     current_text: str,
                                     delta_text: str,
                                     previous_token_ids: List[int],
                                     current_token_ids: List[int],
                                     delta_token_ids: List[int],
                                     ) -> DeltaMessage | None:

        # if the tool call token ID is not in the tokens generated so far, append output to contents
        if self.bot_token_id not in current_token_ids:
            return DeltaMessage(content=delta_text)

        # if the tool call token ID IS in the tokens generated so far, that means we're parsing as tool calls now
        else:

            # handle if we detected the BOT token which means the start of tool calling
            if self.bot_token_id in delta_token_ids:
                logger.info('Found bot_token!')

                # if it's the only token, return None, so we don't send a chat completion
                if len(delta_token_ids) == 1:
                    return None


            # for mistral, everything after the BOT token is tool call, not content. If there's content
            #   which I have yet to see, it would HAVE to come BEFORE the BOT token

            # flags for partial JSON parsing (lib uses bit mask)
            #   if the tool name has been sent then allow any incomplete field ELSE allow everything BUT strings
            #   to avoid sending the partial tool name incorrectly
            flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
            try:

                # note this is basically only the way to do this - just make sure your tool arguments will
                #   never be something containing an apostrophe
                parsable_arr = (current_text
                            .replace(self.bot_token, '')  # remove BOT token to get valid json
                            .replace('\'', '"')  # replace mistral single quotes with double for JSON parsing
                            )
                logger.info('parsing: %s', parsable_arr)
                tool_call_arr: List[Dict] = partial_json_parser.loads(parsable_arr, flags)
                #print('parsed ', tool_call_arr)

                # case: we are starting a new tool in the array
                #   -> array has nonzero length AND length has moved past cursor
                if len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                    # re-set stuff pertaining to progress in the current tool
                    self.current_tool_id = len(tool_call_arr) - 1
                    self.current_tool_name_sent = False
                    self.current_tool_initial_sent = False
                    logger.info('starting on new tool %d', self.current_tool_id)

                # case: update an existing tool
                elif len(tool_call_arr) - 1 == self.current_tool_id and self.current_tool_id >= 0:
                    # logger.info('update to tool %d', self.current_tool_id)
                    pass

                # if there is NOTHING in the array
                else:
                    logger.info('No tool call detected yet!')
                    return None

                # handle parsing
                current_tool_call: Dict = tool_call_arr[self.current_tool_id]

                # if the current tool initial data incl. the id, type=function and idx not sent, send that
                if not self.current_tool_initial_sent:
                    logger.info('Sending InitialDeltaToolCall')
                    self.current_tool_initial_sent = True
                    delta = DeltaMessage(
                        tool_calls=[
                            InitialDeltaToolCall(index=self.current_tool_id).model_dump(exclude_none=True)]
                    )

                # if the current tool name hasn't been sent, send if available - otherwise no chunks
                elif not self.current_tool_name_sent:
                    function_name = current_tool_call.get('name')
                    if function_name:
                        logger.info(f'Sending DeltaToolCall with function name {function_name}!')
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True))
                        ])
                        self.current_tool_name_sent = True
                    else:
                        delta = None

                # now we know we're on the same tool call and we're streaming arguments
                else:

                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get('arguments')
                    cur_arguments = current_tool_call.get('arguments')

                    if not cur_arguments and not prev_arguments:
                        logger.info(f'Skipping text {delta_text} (tokens {delta_token_ids}) - no arguments yet')
                        delta = None
                    elif not cur_arguments and prev_arguments:
                        logger.error('INVARIANT - impossible to have arguments reset mid-arguments')
                        delta = None
                    elif cur_arguments and not prev_arguments:
                        logger.info('First tokens in arguments received')
                        cur_arguments_json = json.dumps(cur_arguments)
                        logger.info(f'Finding {delta_text} in |{cur_arguments_json}')
                        arguments_delta = cur_arguments_json[:cur_arguments_json.index(delta_text) + len(delta_text)]
                        logger.info(f'First tokens in arguments received: {arguments_delta}')
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(
                                arguments=arguments_delta
                            ).model_dump(exclude_none=True))
                        ])

                    elif cur_arguments and prev_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        prev_args_json = json.dumps(prev_arguments)
                        shared_prefix = find_common_prefix(cur_args_json, prev_args_json)
                        cur_args_json = cur_args_json.replace(shared_prefix, '', 1)
                        argument_diff = cur_args_json[:cur_args_json.index(delta_text) + len(delta_text)]
                        logger.info(f'got arguments diff: {argument_diff}')
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(
                                arguments=argument_diff
                            ).model_dump(exclude_none=True))
                        ])
                    else:
                        delta = None

                # check to see if the name is defined and has been sent. if so, stream the name - otherwise keep waiting
                # finish by setting old and returning None as base case
                self.prev_tool_call_arr = tool_call_arr
                return delta

            except Exception as e:
                logger.error(f'Error trying to handle streaming tool call: {e}')
                logger.info('skipping returning a chunk here - maybe we just need more?')
                return None


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

    def __init__(self):
        super().__init__()
        self.current_tool_count: int = 0
        self.current_tool_name_sent: bool = False  # reset each time we encounter a new tool in the array

    def extract_tool_calls_streaming(self,
                                     previous_text: str,
                                     current_text: str,
                                     delta_text: str,
                                     previous_token_ids: List[int],
                                     current_token_ids: List[int],
                                     delta_token_ids: List[int]
                                     ) -> DeltaMessage:
        raise NotImplementedError('Hermes2ProToolParser.extract_tool_calls_streaming has not been implemented!')
