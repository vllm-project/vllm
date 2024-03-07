import os
import json
import jinja2
from enum import Enum
from typing import List, Union
from vllm.logger import init_logger
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest, ChatCompletionToolParam,
                       ChoiceDeltaToolCall, ChatCompletionMessageToolCall,
                       Function, ChatCompletionAssistantMessage,
                       ChatCompletionToolMessage, ChatCompletionNamedToolChoiceParam)

logger = init_logger(__name__)


class ToolsCallsTemplateContext(Enum):
    """ This is used within the template to generate depending on the context. """
    CALL_TOKEN = 1
    FUNCTIONS_LIST = 2
    FORCE_CALL = 3
    CALLS_NOTIF = 4
    TOOL_RESPONSE = 5


class ToolsCallsTemplate:
    """ This template system is only used when the tool_choice is set to "auto" """

    def __init__(self, template_path=None):
        self.trim_blocks = True
        self.lstrip_blocks = True
        if template_path is None:
            template_path = os.path.dirname(
                __file__) + "/templates/tools_functions.jinja"
        self.environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.dirname(template_path)))
        self.template = self.environment.get_template(
            os.path.basename(template_path))
        self.template.globals[
            "FUNCTIONS_LIST"] = ToolsCallsTemplateContext.FUNCTIONS_LIST
        self.template.globals[
            "FORCE_CALL"] = ToolsCallsTemplateContext.FORCE_CALL
        self.template.globals[
            "CALL_TOKEN"] = ToolsCallsTemplateContext.CALL_TOKEN
        self.template.globals[
            "CALLS_NOTIF"] = ToolsCallsTemplateContext.CALLS_NOTIF
        self.template.globals[
            "TOOL_RESPONSE"] = ToolsCallsTemplateContext.TOOL_RESPONSE

    def get_func_call_token(self) -> str:
        """ Return the special token used to find functions calls. """
        return self.template.render(
            CONTEXT=ToolsCallsTemplateContext.CALL_TOKEN)

    def render_toolcalls(self, tool_calls: [ChatCompletionMessageToolCall]):
        return self.template.render(
            CONTEXT=ToolsCallsTemplateContext.CALLS_NOTIF,
            tool_calls=tool_calls)

    def render_toolmessage(self, message: ChatCompletionToolMessage):
        return self.template.render(
            CONTEXT=ToolsCallsTemplateContext.TOOL_RESPONSE, message=message)

    def render_toolslist(self, tool_choice: Union[str, ChatCompletionNamedToolChoiceParam],
                         tools_list: [ChatCompletionToolParam]) -> str:
        if isinstance(tool_choice, str) and tool_choice == "auto":
            tool_choice = None
        if tool_choice is not None:
            for tool in tools_list:
                # Search if the tool_choice is in the tools_list
                if tool.type == "function" and tool.function.name == tool_choice:
                    return self.template.render(
                        CONTEXT=ToolsCallsTemplateContext.FORCE_CALL,
                        tool=tool)
            return ""  # Tool not found. What should we do ?
        else:
            return self.template.render(
                CONTEXT=ToolsCallsTemplateContext.FUNCTIONS_LIST,
                tools_list=tools_list)


class OpenAIToolsPrompter:
    """
    https://platform.openai.com/docs/assistants/tools
    """

    def __init__(self, template_path=None):
        self.template = ToolsCallsTemplate(template_path)
        self.call_token_str = self.template.get_func_call_token()
        if self.call_token_str is None:
            logger.error("There is something wrong with the tools template.")
        else:
            self.call_token_pre = self.call_token_str[0]

    def func_call_token_pre(self) -> str:
        return self.call_token_pre

    def func_call_token_size(self) -> int:
        return len(self.call_token_str)

    def func_call_token(self) -> str:
        return self.call_token_str

    def content_from_assistant(self,
                               message: ChatCompletionAssistantMessage) -> str:
        text = self.template.render_toolcalls(message.tool_calls)
        if message.content is None:
            return text
        else:
            return message.content + "\n" + text

    def content_from_tool(self, message: ChatCompletionToolMessage) -> str:
        return self.template.render_toolmessage(message)

    def inject_prompt(self, request: ChatCompletionRequest):
        """ Generate and inject the prompt for tools calls. """
        if request.tools is not None and self.call_token_str is not None and len(
                request.tools):
            if (isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)):
                if request.tool_choice.type == "function":
                    select_tool_choice = request.tool_choice.function.name
                else:
                    select_tool_choice = None
            else:
                select_tool_choice = None
            text_inject = self.template.render_toolslist(
            tool_choice=select_tool_choice, tools_list=request.tools)
            if isinstance(request.messages, str):
                request.messages = text_inject + request.messages
            elif isinstance(request.messages,
                            List) and len(request.messages) >= 1:
                request.messages[
                    0].content = text_inject + request.messages[0].content


class ChatPromptCapture:

    def __init__(self):
        self.content: str = ""
        self.maybe_function_call = False
        self.is_function_call = False
        self.prefix_size = 0
        self.calls_list: List[{}] = []
        self.after_new_function_call = False
        self.named_function_call = False  # True if the function call is forced using request.tool_choice

    def __str__(self):
        """ Show current state. For debugging purpose. """
        return (f"ChatPromptCapture {{\n"
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

    def num_calls(self):
        return len(self.calls_list)

    def startNamedFunction(self, tool_choice: ChatCompletionNamedToolChoiceParam):
        # Shound not have to be templated since it's defined by the OpenAI reference:
        self.content = "{ \"name\": \"" + tool_choice.function.name + "\", \"arguments\": "
        self.named_function_call = True
        self.prefix_size = 0
        self.is_function_call = True

    def closeNamedFunction(self):
        self.content += "}"

    def checkBracketsFunctionCall(self) -> bool:
        """ Count brackets in a string to check if the function call is complete. """
        if self.content.rfind("}", -6) != -1:
            c1 = self.content.count("{")
            c2 = self.content.count("}")
            if self.named_function_call:
                return c1 == (c2 + 1)
            else:
                return c1 == c2  # We have the complete call block

    def make_calls_list(self, prompter: OpenAIToolsPrompter):
        """ Convert the extracted text to json function calls. """
        if self.named_function_call:
            try:
                call_dict = json.loads(self.content)
                if "name" in call_dict:
                    self.calls_list.append(call_dict)
            except json.decoder.JSONDecodeError as exc:
                # Simply ignore invalid functions calls...
                logger.warning("Error in parsing the function call. This should not happen since it's guided generation : %s" % str(exc))
                pass
        else:
            calls_list = self.content.split(prompter.func_call_token())
            for v_call in calls_list:
                try:
                    call_dict = json.loads(v_call)
                    if "name" in call_dict:
                        self.calls_list.append(call_dict)
                except json.decoder.JSONDecodeError:
                    # Simply ignore invalid functions calls...
                    pass

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
            return ChatCompletionMessageToolCall(index=call_id, id="call_" + call["name"] +
                                                 "_" + str(call_id),
                                                 type="function",
                                                 function=function_call)
        return None

    def to_ChatCompletionMessageToolCallList(
            self) -> [ChatCompletionMessageToolCall]:
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

    def to_ChoiceDeltaToolCallList(self):
        calls_count = self.num_calls()
        tools_calls_list = []
        for call_id in range(calls_count):
            tools_calls_list.append(
                self.to_ChoiceDeltaToolCall(call_id=call_id))
        return tools_calls_list
