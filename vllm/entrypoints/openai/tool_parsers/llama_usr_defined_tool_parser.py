# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Dict, List, Sequence, Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)

def _count_substring(string, substring):
        """
        Counts the number of non-overlapping occurrences of a substring in a string.
        
        Args:
            string (str): The string to search in.
            substring (str): The substring to search for.
            
        Returns:
            int: The number of non-overlapping occurrences of the substring in the string.
        """
        count = 0
        start = 0
        while True:
            start = string.find(substring, start)
            if start == -1:
                break
            count += 1
            start += len(substring)
        return count

@ToolParserManager.register_module("llama3_user_defined_custom")
class Llama3UserDefinedCustomToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if isinstance(self.model_tokenizer, MistralTokenizer):
            logger.error(
                "Detected Mistral tokenizer when using a Llama model")
            self.model_tokenizer = self.model_tokenizer.tokenizer

        self.prev_tool_call_arr: List[Dict] = []
        self.streamed_args_for_tool: List[str] = []
        self.is_parsing_toolcall = False
        
        self.nb_tool_calls = 0
        self.current_tool_name=""
        self.current_tool_call_uuid=""
        self.is_current_tool_name_sent = False
        self.tool_call_start_token: str = "<function"
        self.tool_call_precall_token: str = '>{"'
        self.tool_call_end_token: str = "</function>"
        self.bot_token = "<|python_tag|>"

        self.tool_call_start_token_id = tokenizer.encode(self.tool_call_start_token,
                                             add_special_tokens=False)
        
        self.tool_call_end_token_id = tokenizer.encode(self.tool_call_end_token,
                                             add_special_tokens=False)
          
        self.tool_call_preargs_token_id = tokenizer.encode(self.tool_call_precall_token,
                                             add_special_tokens=False)   
                                            
        self.bot_token_id = tokenizer.encode(self.bot_token,
                                             add_special_tokens=False)

        self.tool_call_regex = re.compile(r"<function=([^>]+)>\{([^}]+)\}(?:</function>|>)?")

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")
        
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
                function_call_tuples = self.tool_call_regex.findall(model_output)
                
                logger.info("function_call_tuples: %s", function_call_tuples)
                print("function_call_tuples: %s", function_call_tuples)
                
                # load the JSON, and then use it to build the Function and
                # Tool Call
                raw_function_calls = [
                    {
                        "name":match[0],
                        "arguments":json.loads("{"+match[1]+"}")
                     } 
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
        """
        Extract tool calls from a streaming response.
        Handles format: <function=functionName{arguments}>
        Returns DeltaMessage with either tool_calls or content.
        """
        logger.debug("\n" + "="*50)
        logger.debug("STREAMING FUNCTION CALLED")
        logger.debug("Tool call start token id IDs:", self.tool_call_start_token_id)
        logger.debug("Tool call precall token id IDs:", self.tool_call_preargs_token_id)
        logger.debug("Tool call end token id IDs:", self.tool_call_end_token_id)
        logger.debug("Previous text:", previous_text)
        logger.debug("Current text:", current_text)
        logger.debug("Delta text:", delta_text)
        logger.debug("Previous token IDs:", previous_token_ids)
        logger.debug("Current token IDs:", current_token_ids)
        logger.debug("Delta token IDs:", delta_token_ids)
        logger.debug("Current tool name sent:", self.is_current_tool_name_sent)
        logger.debug("-"*50 + "\n")
        flags = Allow.ALL if self.is_current_tool_name_sent \
                else Allow.ALL & ~Allow.STR
        
        logger.debug(f"{delta_token_ids[0] in self.tool_call_start_token_id=}")
        if delta_token_ids[0] in self.tool_call_start_token_id : 
            # We possibly have a tool call (not sure yet) we don't stream
          
            logger.debug(f"{_count_substring(current_text,self.tool_call_start_token)=}")
            if _count_substring(current_text,self.tool_call_start_token) > self.nb_tool_calls \
                and not self.is_parsing_toolcall :

                self.is_parsing_toolcall=True
                self.nb_tool_calls +=1 #will serve as id
                self.current_tool_call_uuid = random_uuid()
                logger.debug("New tool call detected, id:", self.nb_tool_calls-1)
                return None # going to the next iter 
            else : 
                logger.debug("Tool call already parsed, id:", self.nb_tool_calls-1)
            
        if self.is_parsing_toolcall and not self.is_current_tool_name_sent : 
            logger.debug("Parsing tool call, id:", self.nb_tool_calls-1)
            # We are parsing a tool call, we need to parse the tool name
            if delta_token_ids != self.tool_call_preargs_token_id:
                self.current_tool_name += delta_text
                logger.debug(f"{self.current_tool_name=}")
                return None # moving on to the next iteration
            else : 
                self.current_tool_name = self.current_tool_name.lstrip('=')
                self.is_current_tool_name_sent = True
                return DeltaMessage(tool_calls=[
                    DeltaToolCall(index=self.nb_tool_calls - 1,
                                    type="function",
                                    id=f"chatcmpl-tool-{self.current_tool_call_uuid}",
                                    function=DeltaFunctionCall(
                                        name=self.current_tool_name))
                ])
            
        if self.is_current_tool_name_sent :
            logger.debug("Parsed tool name : ", self.current_tool_name)

            if _count_substring(current_text,self.tool_call_end_token) < self.nb_tool_calls:
                self.streamed_args_for_tool.append(delta_text)
                return None # moving on to the next iteration
            else :
                arguments = '{"'+''.join(self.streamed_args_for_tool) # adding back {" at the beginning for valid JSON
                arguments = arguments.rstrip(self.tool_call_end_token) # removing the end token
                logger.debug("Concatenated tool call arguments  : ", arguments)

                current_tool_args = partial_json_parser.loads(
                arguments or "{}",
                flags) if self.streamed_args_for_tool else None
                
                logger.debug("Parsed tool call arguments : ", current_tool_args)

                
                delta = DeltaMessage(tool_calls=[
                    DeltaToolCall(index=self.nb_tool_calls - 1,
                                    type="function",
                                    id=f"chatcmpl-tool-{self.current_tool_call_uuid}",
                                    function=DeltaFunctionCall(
                                        name=self.current_tool_name,
                                        arguments=json.dumps(current_tool_args)))
                ])

                self.reset_state()
                
                return delta 
        else : 
            logger.debug("No tool call detected, returning just text : ", delta_text)
            return DeltaMessage(content=delta_text)
            
    def reset_state(self):
        self.current_tool_name = ''
        self.is_parsing_toolcall=False
        self.is_current_tool_name_sent = False
        self.streamed_args_for_tool = []