import ast
import json
import keyword
import re
import traceback
from typing import Dict, List, Sequence, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.logger import init_logger

logger = init_logger(__name__)


@ToolParserManager.register_module("minicpm")
class MiniCPMToolParser(ToolParser):
    """
    Tool call parser for MiniCPM3 4B models intended for use with the
    examples/tool_chat_template_minicpm3.jinja template.

    Used when --enable-auto-tool-choice --tool-call-parser minicpm are all set
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.thought_start_token = "<|thought_start|>"
        self.thought_end_token = "<|thought_end|>"
        self.tool_call_start_token = "<|tool_call_start|>"
        self.tool_call_end_token = "<|tool_call_end|>"
        self.stop_token_ids = [2, 73440]

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """
        msg = fc2dict(model_output)
        if ("tool_calls" in msg and msg["tool_calls"] is not None
                and len(msg["tool_calls"]) > 0):
            tool_calls: List[ToolCall] = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=raw_function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(raw_function_call["arguments"],
                                             ensure_ascii=False),
                    ),
                ) for raw_function_call in msg["tool_calls"]
            ]

            # get any content before  the tool call
            ret = ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=msg.get("content", None),
            )
            return ret
        else:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[],
                content=msg.get("content", None),
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
        # if no tools are provided, we don't need to parse tool calls
        if not request.tools:
            return DeltaMessage(content=delta_text)
        if self.thought_end_token not in current_text:
            return None
        useful_text = current_text.split(self.thought_end_token)[1]
        if (current_token_ids[-1]
                in self.stop_token_ids):  # case 3: stream generation ended
            msg = fc2dict(current_text)
            if ("tool_calls" in msg and msg["tool_calls"] is not None
                    and len(msg["tool_calls"]) > 0):
                self.prev_tool_call_arr = msg["tool_calls"]
                self.streamed_args_for_tool = ["" for tc in msg["tool_calls"]]
                delta_message = DeltaMessage(
                    role="assistant",
                    content=msg.get("content", None),
                )
                return delta_message
            else:
                return DeltaMessage(content=msg.get("content", None))
        elif (self.tool_call_start_token in useful_text
              and self.tool_call_end_token
              in useful_text):  # case 2: tool call ended
            return None
        elif (self.tool_call_start_token
              in useful_text):  # case 1: tool call started
            # Extract function name and arguments, handling nested parentheses
            pattern = r"(\w+)\(((?:[^()]*|\([^()]*\))*)\)"
            matches = re.finditer(pattern, useful_text)
            tool_calls: List[Dict] = []
            delta = None
            for idx, match in enumerate(matches):
                if self.current_tool_id < idx:
                    self.current_tool_id = idx
                    func_name = match.group(1)
                    func_args = match.group(2)
                    tool_call_string = f"{func_name}({func_args})\n"

                    parsed = ast.parse(tool_call_string)
                    for elem in parsed.body:
                        assert isinstance(elem.value, ast.Call)  # type: ignore
                        calls = resolve_ast_call(elem.value)  # type: ignore

                        for func_name, func_args in calls.items():
                            this_call = {
                                "name":
                                func_name,
                                "arguments":
                                json.dumps(func_args, ensure_ascii=False),
                            }
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    **this_call).model_dump(exclude_none=True),
                            )
                        ])
            self.prev_tool_call_arr = tool_calls
            self.streamed_args_for_tool = ["" for x in tool_calls]
            self.current_tool_name_sent = True
            return delta
        else:
            return None


def fc2dict(
    sequence: str,
    tool_call_start="<|tool_call_start|>",
    tool_call_end="<|tool_call_end|>",
    thought_start="<|thought_start|>",
    thought_end="<|thought_end|>",
):
    if thought_end in sequence and thought_start in sequence:
        thought_string, sequence = sequence.rsplit(thought_end, 1)
        thought_string = thought_string.split(thought_start, 1)[1]
    else:
        thought_string = ""
    if tool_call_start in sequence and tool_call_end in sequence:
        tool_call_string, content = sequence.rsplit(tool_call_end, 1)
        tool_call_string = tool_call_string.split(tool_call_start, 1)[1]
        try:
            tool_calls = []
            tool_call_string = tool_call_string.strip()
            if tool_call_string.startswith("```"):
                tool_call_string = tool_call_string[3:].strip()
                if tool_call_string.startswith("python"):
                    tool_call_string = tool_call_string.lstrip(
                        "python").strip()
            if tool_call_string.endswith("```"):
                tool_call_string = tool_call_string[:-3].strip()
            for kw in keyword.kwlist:
                tool_call_string = tool_call_string.replace(
                    "," + kw + "=", "," + kw + "_=")
                tool_call_string = tool_call_string.replace(
                    " " + kw + "=", " " + kw + "_=")
                tool_call_string = tool_call_string.replace(
                    "(" + kw + "=", "(" + kw + "_=")

            parsed: ast.Module = ast.parse(tool_call_string)

            for elem in parsed.body:
                assert isinstance(elem.value, ast.Call)  # type: ignore
                calls = resolve_ast_call(elem.value)  # type: ignore

                for func_name, func_args in calls.items():
                    new_args = {}
                    for k, v in func_args.items():
                        for kw in keyword.kwlist:
                            if k == kw + "_":
                                k = kw
                        new_args[k] = v

                    this_one = {"name": func_name, "arguments": new_args}
                    tool_calls.append(this_one)

            return {
                "content": content.strip(),
                "tool_calls": tool_calls,
                "role": "assistant",
            }
        except Exception as e:
            logger.error("Error parsing tool call: %s", str(e))
            logger.error(traceback.format_exc())
            return {
                "content": content.strip(),
                "role": "assistant",
                "thought": thought_string,
            }
    else:
        return {
            "content": sequence.strip(),
            "role": "assistant",
            "thought": thought_string,
        }


# from ShishirPatil/gorilla
def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        output = "..." if value.value is Ellipsis else value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value  # type: ignore
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
            value,
            ast.NameConstant):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
            value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = ast.literal_eval(ast.unparse(value))  # type: ignore
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)  # type: ignore
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = ast.literal_eval(
            ast.unparse(  # type: ignore
                value.body[0].value))  # type: ignore
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)  # type: ignore
        except Exception as e:
            logger.error("Error parsing tool call: %s", str(e))
            output = (
                ast.unparse(value.value) + "[" +  # type: ignore
                ast.unparse(value.slice) + "]")  # type: ignore
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output
