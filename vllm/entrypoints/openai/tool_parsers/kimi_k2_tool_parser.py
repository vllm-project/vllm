# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# code modified from deepseekv3_tool_parser.py

from collections.abc import Sequence
from typing import Union
import json
import regex as re

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


def normalize_quotes_for_json(text: str) -> str:
    """
    Normalize mixed quotes in text to make it valid JSON.
    Handles cases like: '[{\'type\': \'text\', \'text\': "[{\'type\': \'text\', \'text\': \'content\'}]"}]'
    Also removes unnecessary nested [{"type": "text", "text": "..."}] structures.
    """
    if not text or not isinstance(text, str):
        return text
    
    def unwrap_nested_text_structure(s: str, max_depth: int = 10) -> str:
        """
        Recursively unwrap nested [{"type": "text", "text": "..."}] structures.
        Returns the innermost meaningful content.
        """
        if max_depth <= 0:
            return s  # Prevent infinite recursion
            
        try:
            # First try to parse as-is
            parsed = json.loads(s)
        except json.JSONDecodeError:
            # If parsing fails, try normalizing quotes first
            try:
                normalized = normalize_json_quotes(s)
                parsed = json.loads(normalized)
            except:
                return s  # If still fails, return as-is
                
        # Check if this is a nested text structure
        if (isinstance(parsed, list) and len(parsed) == 1 and 
            isinstance(parsed[0], dict) and 
            parsed[0].get("type") == "text" and 
            "text" in parsed[0]):
            
            inner_text = parsed[0]["text"]
            if isinstance(inner_text, str) and inner_text.strip():
                # If inner text looks like more JSON, unwrap recursively
                if inner_text.strip().startswith(('[', '{')):
                    return unwrap_nested_text_structure(inner_text, max_depth - 1)
                else:
                    # Return the plain text content
                    return inner_text
        return s  # Not a nested structure, return as-is
    
    def normalize_json_quotes(s: str) -> str:
        """Normalize quotes in a JSON-like string using state machine approach."""
        result = []
        in_string = False
        escape_next = False
        string_start_char = None
        i = 0
        
        while i < len(s):
            char = s[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
            elif char == '\\':
                result.append(char)
                escape_next = True
            elif not in_string:
                if char == "'" or char == '"':
                    result.append('"')
                    in_string = True
                    string_start_char = char
                else:
                    result.append(char)
            else:
                if char == string_start_char:
                    result.append('"')
                    in_string = False
                    string_start_char = None
                elif char == '"' and string_start_char == "'":
                    result.append('\\"')
                elif char == "'" and string_start_char == '"':
                    result.append("'")
                else:
                    result.append(char)
            i += 1
        
        return ''.join(result)
    
    # First, try to unwrap nested structures
    unwrapped = unwrap_nested_text_structure(text)
    
    # If unwrapping produced a different result, return it
    if unwrapped != text:
        # If the unwrapped content is plain text, return it directly
        if not unwrapped.strip().startswith(('[', '{')):
            # Check if the content is empty or meaningless
            if not unwrapped or unwrapped.strip() == '':
                return ''  # Return empty string for empty content
            return unwrapped
        # If it's still JSON-like, normalize it
        return normalize_json_quotes(unwrapped)
    
    # Handle mixed content (JSON + tool call markers)
    if any(marker in text for marker in ['<|tool_call', '<|tool_calls']):
        return normalize_json_quotes(text)
    
    # For pure JSON, try to parse and return as-is if valid
    stripped = text.strip()
    if stripped.startswith(('[', '{')):
        try:
            json.loads(text)
            return text  # Already valid JSON
        except json.JSONDecodeError:
            # Invalid JSON, normalize quotes
            return normalize_json_quotes(text)
    
    # For non-JSON text, return as-is
    return text


@ToolParserManager.register_module(["kimi_k2"])
class KimiK2ToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = (
            [])  # map what has been streamed for each tool so far to a list

        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>.+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<tool_call_id>.+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)"
        )

        self.stream_tool_call_name_regex = re.compile(
            r"(?P<tool_call_id>.+:\d+)\s*")

        # New regexes for the updated format (function.name:id)
        self.stream_tool_call_portion_regex_new = re.compile(
            r"(?P<tool_call_id>function\.[^:]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)"
        )

        self.stream_tool_call_name_regex_new = re.compile(
            r"(?P<tool_call_id>function\.[^:]+:\d+)\s*")

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")
        self.tool_calls_start_token_id = self.vocab.get(
            self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(
            self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (self.tool_calls_start_token_id is None
                or self.tool_calls_end_token_id is None):
            raise RuntimeError(
                "Kimi-K2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")


    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        # Check for both standard format and Kimi format
        # Standard: <|tool_calls_section_begin|><|tool_call_begin|>functions.name:id<|tool_call_argument_begin|>...<|tool_call_end|><|tool_calls_section_end|>
        # Kimi: <|tool_call_end|><|tool_call_begin|>functions.name:id<|tool_call_argument_begin|>...<|tool_call_end|><|tool_calls_section_end|>
        
        try:
            # Use a comprehensive regex that handles both formats
            # Support both "function." and "functions." prefixes
            # Capture the prefix to preserve it in the output
            tool_call_pattern = re.compile(
                r"<\|tool_call_begin\|>\s*(functions?)\.([^:]+):(\d+)\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
                re.DOTALL
            )
            matches = tool_call_pattern.findall(model_output)
            
            if matches:
                tool_calls = []
                for prefix, function_name, call_id, function_args in matches:
                    # # Normalize quotes in function arguments to ensure valid JSON
                    # normalized_args = normalize_quotes_for_json(function_args)
                    normalized_args = function_args
                    tool_calls.append(
                        ToolCall(
                            id=f"{prefix}.{function_name}:{call_id}",
                            type='function',
                            function=FunctionCall(name=function_name,
                                                  arguments=normalized_args),
                        ))

                # Extract content before tool calls (following兜底逻辑 approach)
                # Always extract content before <|tool_call_begin|>, then clean misplaced markers
                content = ""
                first_tool_begin = model_output.find(self.tool_call_start_token)
                if first_tool_begin != -1:
                    content = model_output[:first_tool_begin]
                    
                    # Remove misplaced <|tool_call_end|> markers that should have been <|tool_calls_section_begin|>
                    # This handles cases where the model replaced the section begin marker
                    content = re.sub(r"<\|tool_call_end\|>", "", content)
                    
                    # Clean up any remaining markers
                    content = re.sub(r"<\|tool_calls_section_end\|>\s*$", "", content, flags=re.DOTALL)
                    
                    # Clean other markers
                    markers_to_clean = [
                        self.tool_calls_start_token,
                        self.tool_calls_end_token,
                        "<|tool_call_argument_begin|>"
                    ]
                    for marker in markers_to_clean:
                        content = content.replace(marker, "")
                    content = content.strip()
                    
                    # # Normalize quotes in content to ensure valid JSON
                    # if content:
                    #     content = normalize_quotes_for_json(content)

                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

        except Exception:
            logger.exception("Error in extracting tool calls from response.")
            
        # If no tool calls found, return cleaned content
        cleaned_content = model_output
        markers_to_clean = [
            self.tool_calls_start_token,
            self.tool_calls_end_token,
            self.tool_call_start_token, 
            self.tool_call_end_token,
            "<|tool_call_argument_begin|>"
        ]
        for marker in markers_to_clean:
            cleaned_content = cleaned_content.replace(marker, "")
        cleaned_content = cleaned_content.strip()
        
        # Normalize quotes in content to ensure valid JSON
        # if cleaned_content:
        #     cleaned_content = normalize_quotes_for_json(cleaned_content)
        
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=cleaned_content if cleaned_content else None
        )
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

        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        # Check if we should be streaming a tool call
        # Standard format: has <|tool_calls_section_begin|>
        # Kimi format: has <|tool_call_begin|>, <|tool_call_end|>, or <|tool_calls_section_end|>
        has_section_begin = self.tool_calls_start_token_id in current_token_ids
        has_tool_call_tokens = (self.tool_call_start_token_id in current_token_ids or 
                               self.tool_call_end_token_id in current_token_ids or
                               self.tool_calls_end_token_id in current_token_ids)
        
        if not has_section_begin and not has_tool_call_tokens:
            logger.debug("No tool call tokens found!")
            # Normalize quotes in content to ensure valid JSON
            normalized_content = normalize_quotes_for_json(delta_text) if delta_text else delta_text
            return DeltaMessage(content=normalized_content)
        delta_text = delta_text.replace(self.tool_calls_start_token,
                                        "").replace(self.tool_calls_end_token,
                                                    "")
        try:

            # figure out where we are in the parsing by counting tool call
            # start & end tags
            prev_tool_start_count = previous_token_ids.count(
                self.tool_call_start_token_id)
            prev_tool_end_count = previous_token_ids.count(
                self.tool_call_end_token_id)
            cur_tool_start_count = current_token_ids.count(
                self.tool_call_start_token_id)
            cur_tool_end_count = current_token_ids.count(
                self.tool_call_end_token_id)
            tool_call_portion = None
            text_portion = None

            # Special case: handle Kimi format where end token appears before start token
            malformed_message = False
            if (cur_tool_start_count == cur_tool_end_count 
                    and cur_tool_start_count > 0
                    and self.tool_call_end_token in current_text):
                # Check if end token comes before start token
                first_end_pos = current_text.find(self.tool_call_end_token)
                first_start_pos = current_text.find(self.tool_call_start_token)
                if first_end_pos != -1 and first_start_pos != -1 and first_end_pos < first_start_pos:
                    logger.debug("Detected Kimi format with end token before start token - this is normal")
                    malformed_message = True
                    # Continue with normal tool call processing instead of treating as text
                elif first_end_pos != -1 and first_start_pos == -1:
                    # Only treat as stray if we don't have section end token (which indicates Kimi format)
                    if self.tool_calls_end_token not in current_text:
                        logger.debug("Detected stray end token without matching start, treating as text")
                        # Normalize quotes in content to ensure valid JSON
                        normalized_content = normalize_quotes_for_json(delta_text) if delta_text else delta_text
                        return DeltaMessage(content=normalized_content)
                    else:
                        logger.debug("Detected Kimi format with end token before start - continuing processing")
                        malformed_message = True
            
            # case: if we're generating text, OR rounding out a tool call
            # But skip this check for malformed messages that need special handling
            if (not malformed_message 
                    and cur_tool_start_count == cur_tool_end_count
                    and prev_tool_end_count == cur_tool_end_count
                    and self.tool_call_end_token not in delta_text
                    and cur_tool_start_count > 0):
                logger.debug("Generating text content! skipping tool parsing.")
                # Normalize quotes in content to ensure valid JSON
                normalized_content = normalize_quotes_for_json(delta_text) if delta_text else delta_text
                return DeltaMessage(content=normalized_content)

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text + delta_text
                tool_call_portion = full_text.split(
                    self.tool_call_start_token)[-1].split(
                        self.tool_call_end_token)[0].rstrip()
                delta_text = delta_text.split(
                    self.tool_call_end_token)[0].rstrip()
                text_portion = delta_text.split(
                    self.tool_call_end_token)[-1].lstrip()

            # case -- we're starting a new tool call
            if (cur_tool_start_count > cur_tool_end_count
                    and cur_tool_start_count > prev_tool_start_count):
                if len(delta_token_ids) > 1:
                    tool_call_portion = current_text.split(
                        self.tool_call_start_token)[-1]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None

                # set cursors and state appropriately
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # case -- we're updating an existing tool call
            elif (cur_tool_start_count > cur_tool_end_count
                  and cur_tool_start_count == prev_tool_start_count):

                # get the portion of the text that's the tool call
                tool_call_portion = current_text.split(
                    self.tool_call_start_token)[-1]
                text_portion = None

            # case -- the current tool call is being closed.
            elif (cur_tool_start_count == cur_tool_end_count
                  and cur_tool_end_count >= prev_tool_end_count):
                if self.prev_tool_call_arr is None or len(
                        self.prev_tool_call_arr) == 0:
                    logger.debug(
                        "attempting to close tool call, but no tool call")
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments")
                if diff:
                    diff = (diff.encode("utf-8").decode("unicode_escape")
                            if diff is str else diff)
                    if '"}' not in delta_text:
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    
                    # Clean tool call markers from final diff
                    clean_diff = diff
                    for marker in [self.tool_call_end_token, self.tool_calls_end_token, 
                                  "<|tool_call_argument_begin|>"]:
                        clean_diff = clean_diff.replace(marker, "")
                    
                    logger.debug(
                        "Finishing tool and found diff that had not "
                        "been streamed yet: %s",
                        diff,
                    )
                    if clean_diff != diff:
                        logger.debug("cleaned final diff from %s to %s", 
                                   repr(diff), repr(clean_diff))
                    
                    # Update stored arguments with cleaned version
                    clean_stored_args = self.streamed_args_for_tool[self.current_tool_id] + clean_diff
                    for marker in [self.tool_call_end_token, self.tool_calls_end_token, 
                                  "<|tool_call_argument_begin|>"]:
                        clean_stored_args = clean_stored_args.replace(marker, "")
                    self.streamed_args_for_tool[self.current_tool_id] = clean_stored_args
                    
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=clean_diff).model_dump(exclude_none=True),
                        )
                    ])

            # case -- otherwise we're just generating text
            else:
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                # Normalize quotes in content to ensure valid JSON
                normalized_text = normalize_quotes_for_json(text) if text else text
                delta = DeltaMessage(tool_calls=[], content=normalized_text)
                return delta

            current_tool_call = dict()
            if tool_call_portion:
                # Try new format first (function.name:id)
                current_tool_call_matches = (
                    self.stream_tool_call_portion_regex_new.match(
                        tool_call_portion))
                if current_tool_call_matches:
                    tool_id, tool_args = (current_tool_call_matches.groups())
                    tool_name = tool_id.split('.')[1].split(':')[0]
                    current_tool_call['id'] = tool_id
                    current_tool_call["name"] = tool_name
                    # Normalize quotes in tool arguments to ensure valid JSON
                    current_tool_call["arguments"] = normalize_quotes_for_json(tool_args)
                else:
                    # Try old format (functions.name:id) for backward compatibility
                    current_tool_call_matches = (
                        self.stream_tool_call_portion_regex.match(
                            tool_call_portion))
                    if current_tool_call_matches:
                        tool_id, tool_args = (current_tool_call_matches.groups())
                        # Both formats use the same parsing logic now (both have dot then colon)
                        tool_name = tool_id.split('.')[1].split(':')[0]
                        current_tool_call['id'] = tool_id
                        current_tool_call["name"] = tool_name
                        # Normalize quotes in tool arguments to ensure valid JSON
                        current_tool_call["arguments"] = normalize_quotes_for_json(tool_args)
                    else:
                        # Try name-only patterns - new format first
                        current_tool_call_name_matches = (
                            self.stream_tool_call_name_regex_new.match(
                                tool_call_portion))
                        if current_tool_call_name_matches:
                            tool_id_str, = current_tool_call_name_matches.groups()
                            tool_name = tool_id_str.split('.')[1].split(':')[0]
                            current_tool_call['id'] = tool_id_str
                            current_tool_call["name"] = tool_name
                            current_tool_call["arguments"] = ""
                        else:
                            # Try old name-only pattern
                            current_tool_call_name_matches = (
                                self.stream_tool_call_name_regex.match(
                                    tool_call_portion))
                            if current_tool_call_name_matches:
                                tool_id_str, = current_tool_call_name_matches.groups()
                                tool_name = tool_id_str.split('.')[1].split(':')[0]
                                current_tool_call['id'] = tool_id_str
                                current_tool_call["name"] = tool_name
                                current_tool_call["arguments"] = ""
                            else:
                                logger.debug("Not enough token")
                                return None

            # case - we haven't sent the tool name yet. If it's available, send
            #   it. otherwise, wait until it's available.
            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name: Union[str, None] = current_tool_call.get("name")
                tool_id = current_tool_call.get("id")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=tool_id,
                            function=DeltaFunctionCall(
                                name=function_name).model_dump(
                                    exclude_none=True),
                        )
                    ])
                else:
                    return None

            # case -- otherwise, send the tool call delta

            # if the tool call portion is None, send the delta as text
            if tool_call_portion is None:
                # if there's text but not tool calls, send that -
                # otherwise None to skip chunk
                # Normalize quotes in content to ensure valid JSON
                normalized_content = normalize_quotes_for_json(delta_text) if delta_text else delta_text
                delta = (DeltaMessage(
                    content=normalized_content) if text_portion is not None else None)
                return delta

            # now, the nitty-gritty of tool calls
            # now we have the portion to parse as tool call.

            logger.debug("Trying to parse current tool call with ID %s",
                         self.current_tool_id)

            # if we're starting a new tool call, push an empty object in as
            #   a placeholder for the arguments
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # main logic for tool parsing here - compare prev. partially-parsed
            #   JSON to the current partially-parsed JSON
            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments")
            cur_arguments = current_tool_call.get("arguments")

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            # case -- no arguments have been created yet. skip sending a delta.
            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None

            # case -- prev arguments are defined, but non are now.
            #   probably impossible, but not a fatal error - just keep going
            elif not cur_arguments and prev_arguments:
                logger.error("should be impossible to have arguments reset "
                             "mid-call. skipping streaming anything.")
                delta = None

            # case -- we now have the first info about arguments available from
            #   autocompleting the JSON
            elif cur_arguments and not prev_arguments:
                # Clean tool call markers from initial arguments
                clean_cur_arguments = cur_arguments
                for marker in [self.tool_call_end_token, self.tool_calls_end_token, 
                              "<|tool_call_argument_begin|>"]:
                    clean_cur_arguments = clean_cur_arguments.replace(marker, "")
                
                if clean_cur_arguments != cur_arguments:
                    logger.debug("cleaned initial arguments from %s to %s", 
                               repr(cur_arguments), repr(clean_cur_arguments))

                delta = DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(
                            arguments=clean_cur_arguments).model_dump(
                                exclude_none=True),
                    )
                ])
                self.streamed_args_for_tool[
                    self.current_tool_id] = clean_cur_arguments

            # last case -- we have an update to existing arguments.
            elif cur_arguments and prev_arguments:
                if (isinstance(delta_text, str)
                        and cur_arguments != prev_arguments
                        and len(cur_arguments) > len(prev_arguments)
                        and cur_arguments.startswith(prev_arguments)):
                    delta_arguments = cur_arguments[len(prev_arguments):]
                    
                    # Clean tool call markers from delta_arguments
                    clean_delta_arguments = delta_arguments
                    for marker in [self.tool_call_end_token, self.tool_calls_end_token, 
                                  "<|tool_call_argument_begin|>"]:
                        clean_delta_arguments = clean_delta_arguments.replace(marker, "")
                    
                    logger.debug("got diff %s", delta_text)
                    if clean_delta_arguments != delta_arguments:
                        logger.debug("cleaned delta arguments from %s to %s", 
                                   repr(delta_arguments), repr(clean_delta_arguments))

                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=clean_delta_arguments).model_dump(
                                    exclude_none=True),
                        )
                    ])
                    # Store cleaned arguments
                    clean_cur_arguments = cur_arguments
                    for marker in [self.tool_call_end_token, self.tool_calls_end_token, 
                                  "<|tool_call_argument_begin|>"]:
                        clean_cur_arguments = clean_cur_arguments.replace(marker, "")
                    self.streamed_args_for_tool[
                        self.current_tool_id] = clean_cur_arguments
                else:
                    delta = None

            # handle saving the state for the current tool into
            # the "prev" list for use in diffing for the next iteration
            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[
                    self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.