import json
from typing import List, Dict, Union
from vllm.logger import init_logger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionToolParam, VllmToolsTemplate,
    ChoiceDeltaToolCall, ChatCompletionMessageToolCall, Function,
    ChatCompletionAssistantMessage, ChatCompletionToolMessage,
    ChatCompletionNamedToolChoiceParam)

logger = init_logger(__name__)


class ToolsCallsTemplate:
    """ This template system is only used when the tool_choice is set to "auto" """

    def __init__(self):
        pass

    def render_toolcalls(self, tool_calls: [ChatCompletionMessageToolCall],
                         tool_params: VllmToolsTemplate) -> str:
        instructions = ""
        if len(tool_params.tool_call_notif_noarg_start) or len(
                tool_params.tool_call_notif_noarg_end):
            for call in tool_calls:
                if call.function.arguments is None or len(
                        call.function.arguments) == 0:
                    instructions += tool_params.tool_call_notif_noarg_start + " " + call.id + " " + tool_params.tool_call_notif_noarg_end + "\n"
                else:
                    instructions += tool_params.tool_call_notif_args_start + " " + call.id + " " + tool_params.tool_call_notif_noarg_end + "\n"
        return instructions

    def render_toolresponse(self, message: ChatCompletionToolMessage,
                            tool_params: VllmToolsTemplate) -> str:
        return tool_params.response_token_start + str(message.content) + tool_params.response_token_end + "\n"

    def render_tool(self, tool: ChatCompletionToolParam,
                    tool_params: VllmToolsTemplate) -> str:
        if tool.function.parameters is None or len(
                tool.function.parameters) == 0:
            return f"""{tool_params.tool_token_start}{{"name": "{tool.function.name}",""" \
                   f""""description": "{tool.function.description}", "arguments": null}}{tool_params.tool_token_end}\n"""
        else:
            return f"""{tool_params.tool_token_start}{{"name": "{tool.function.name}",""" \
                   f""""description": "{tool.function.description}", "arguments": {{ {tool.function.parameters} }}}}{tool_params.tool_token_end}\n"""

    def render_toolslist(self, tool_choice: Union[
        str, ChatCompletionNamedToolChoiceParam],
                         tools_list: [ChatCompletionToolParam],
                         tool_params: VllmToolsTemplate) -> str:
        if isinstance(tool_choice, str) and (tool_choice == "auto"
                                             or tool_choice == "none"):
            tool_choice = None
        if tool_choice is not None:  # Guided generation
            for tool in tools_list:
                # Search if the tool_choice is in the tools_list
                if tool.type == "function" and tool.function.name == tool_choice:
                    instructions = tool_params.function_guided + "\n" + self.render_tool(
                        tool, tool_params=tool_params) + "\n"
                    instructions += tool_params.function_call_instruct
                    return instructions
            return ""  # Tool not found. What should we do ?
        else:
            instructions = tool_params.function_list_start + "\n"
            for tool in tools_list:
                instructions += self.render_tool(tool, tool_params=tool_params)
            instructions += "\n" + tool_params.function_list_end + "\n"
            instructions += tool_params.function_call_instruct
            return instructions


class OpenAIToolsPrompter:
    """
    https://platform.openai.com/docs/assistants/tools
    """

    def __init__(self, privileged: bool):
        self.privileged = privileged
        self.template = ToolsCallsTemplate()

    def content_from_assistant(self, message: ChatCompletionAssistantMessage,
                               tool_params: VllmToolsTemplate) -> str:
        text = self.template.render_toolcalls(message.tool_calls,
                                              tool_params=tool_params)
        if message.content is None:
            return text
        else:
            return message.content + "\n" + text

    def content_from_tool(self, message: ChatCompletionToolMessage,
                          tool_params: VllmToolsTemplate) -> str:
        return self.template.render_toolresponse(message,
                                                 tool_params=tool_params)

    def inject_prompt(self, request: ChatCompletionRequest):
        """ Generate and inject the prompt for tools calls. """
        if request.tools is not None and len(request.tools):
            if (isinstance(request.tool_choice,
                           ChatCompletionNamedToolChoiceParam)):
                if request.tool_choice.type == "function":
                    select_tool_choice = request.tool_choice.function.name
                else:
                    select_tool_choice = None
            else:
                select_tool_choice = None
            text_inject = self.template.render_toolslist(
                tool_choice=select_tool_choice,
                tools_list=request.tools,
                tool_params=request.tool_params)
            if isinstance(request.messages, str):
                request.messages = text_inject + "\n\n" + request.messages
            elif isinstance(request.messages,
                            List) and len(request.messages) >= 1:
                request.messages[
                    0].content = text_inject + "\n\n" + request.messages[
                        0].content


class ChatPromptCapture:

    def __init__(self, prompter: OpenAIToolsPrompter,
                 tool_params: VllmToolsTemplate):
        self.content: str = ""
        self.prompter = prompter
        self.maybe_function_call = False
        self.is_function_call = False
        self.prefix_size = 0
        self.calls_list: List[{}] = []
        self.after_new_function_call = False
        self.named_function_call = False  # True if the function call is forced using request.tool_choice
        self.tool_params = tool_params

    def __str__(self):
        """ Show current state. For debugging purpose. """
        return (
            f"ChatPromptCapture {{\n"
            f"    maybe_function_call={self.maybe_function_call},\n"
            f"    is_function_call={self.is_function_call},\n"
            f"    prefix_size={self.prefix_size},\n"
            f"    after_new_function_call={self.after_new_function_call},\n"
            f"    content={self.content},\n"
            f"    calls_list={self.calls_list},\n"
            f"}}")

    def reset(self, reset_calls_list=False):
        self.content = ""
        self.maybe_function_call = False
        self.is_function_call = False
        self.named_function_call = False
        self.prefix_size = 0
        if reset_calls_list:
            self.calls_list = []

    def func_call_token_pre(self) -> str:
        return self.tool_params.call_token_start[0]
        # return self.call_token_pre

    def func_call_token_size(self) -> int:
        return len(self.tool_params.call_token_start)
        # return len(self.call_token_str)

    def func_call_token(self) -> str:
        return self.tool_params.call_token_start
        # return self.call_token_str

    def num_calls(self):
        return len(self.calls_list)

    def startNamedFunction(self,
                           tool_choice: ChatCompletionNamedToolChoiceParam):
        # Should not have to be templated since it's defined by the OpenAI reference:
        self.content = "{ \"name\": \"" + tool_choice.function.name + "\", \"arguments\": "
        self.named_function_call = True
        self.prefix_size = 0
        self.is_function_call = True

    def closeNamedFunction(self):
        self.content += "}"

    def checkBracketsFunctionCall(self,
                                  tool_params: VllmToolsTemplate) -> bool:
        """ Count brackets in a string to check if the function call is complete. """
        if self.named_function_call:
            if self.content.rfind("}", -6) != -1:
                c1 = self.content.count("{")
                c2 = self.content.count("}")
                return c1 == (c2 + 1)
        else:
            content_end = self.content[-(len(tool_params.call_token_end) +
                                         6):].rstrip()
            if tool_params.call_token_end in content_end and content_end.find(
                    "}") != -1:
                c1 = self.content.count("{")
                c2 = self.content.count("}")
                return c1 == c2  # We have the complete call block

    def make_calls_list(self):
        """ Convert the extracted text to json function calls. """
        if self.named_function_call:
            if self._add_calls_list(self.content) == 0:
                return
        else:
            calls_list = self.content.split(self.tool_params.call_token_start)
            for v_call in calls_list:
                if len(self.tool_params.call_token_end):
                    content = v_call.split(self.tool_params.call_token_end)[0]
                else:
                    content = v_call
                self._add_calls_list(content)

    def _add_calls_list(self, content: str) -> int:
        """ Returns the number of added tools calls. """
        count = len(self.calls_list)
        if len(content) > 1:
            try:
                call_data = json.loads(content)
            except json.decoder.JSONDecodeError as exc:
                # Simply ignore invalid functions calls...
                if self.named_function_call:
                    logger.warning(
                        "Error in parsing the function call. This should not happen since it's guided generation : %s"
                        % str(exc))
                else:
                    logger.warning(
                        "Error in parsing the function call. The model have probably generated a wrong synthax : %s" % str(exc))
                return 0
            if isinstance(call_data, List):
                for call_elem in call_data:
                    if isinstance(call_elem, Dict):
                        if "name" in call_elem:
                            self.calls_list.append(call_elem)
            elif isinstance(call_data, Dict):
                if "name" in call_data:
                    self.calls_list.append(call_data)
            if self.prompter.privileged:
                logger.info("Catched tool call : %s" % str(call_data))
        return len(self.calls_list) - count

    def to_ChatCompletionMessageToolCall(
            self, call_id: int) -> Union[ChatCompletionMessageToolCall, None]:
        if len(self.calls_list) and call_id < len(self.calls_list):
            call = self.calls_list[call_id]
            arguments = call["arguments"] if "arguments" in call else None
            if arguments is None and "parameters" in call:
                arguments = call["parameters"]
            function_call = Function(name=call["name"],
                                     arguments=json.dumps(arguments)
                                     if arguments is not None else "")
            return ChatCompletionMessageToolCall(index=call_id,
                                                 id="call_" + call["name"] +
                                                 "_" + str(call_id),
                                                 type="function",
                                                 function=function_call)
        return None

    def to_ChatCompletionMessageToolCallList(
            self) -> List[ChatCompletionMessageToolCall]:
        calls_count = self.num_calls()
        tools_calls_list = []
        for call_id in range(calls_count):
            tools_calls_list.append(
                self.to_ChatCompletionMessageToolCall(call_id=call_id))
        return tools_calls_list

    def to_ChoiceDeltaToolCall(
            self, call_id: int) -> Union[ChoiceDeltaToolCall, None]:
        mesg = self.to_ChatCompletionMessageToolCall(call_id)
        if mesg is not None:
            return ChoiceDeltaToolCall(index=call_id,
                                       id=mesg.id,
                                       type=mesg.type,
                                       function=mesg.function)
        return None

    def to_ChoiceDeltaToolCallList(
            self) -> List[Union[ChoiceDeltaToolCall, None]]:
        calls_count = self.num_calls()
        tools_calls_list = []
        for call_id in range(calls_count):
            tools_calls_list.append(
                self.to_ChoiceDeltaToolCall(call_id=call_id))
        return tools_calls_list
