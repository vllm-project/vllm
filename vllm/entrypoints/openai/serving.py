import time
import json
import asyncio
import codecs
from http import HTTPStatus
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Dict, List, Optional, Tuple, Union
from vllm.logger import init_logger
from vllm.engine.async_llm_engine import AsyncLLMEngine
from .protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, LogProbs,
    ModelCard, ModelList, ModelPermission, UsageInfo, ChatCompletionToolParam,
    ToolCallsDelta, ToolCallsMessage, FunctionCall,
    ChatCompletionAssistantMessage, ChatCompletionToolMessage, ErrorResponse)
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


class OpenAIToolsPrompter:
    """
    https://platform.openai.com/docs/assistants/tools
    """

    def __init__(self):
        pass

    @classmethod
    def func_call_token_pre(cls) -> str:
        return "!"

    @classmethod
    def func_call_token_size(cls) -> int:
        return 15

    @classmethod
    def func_call_token(cls) -> str:
        return "!function_call:"

    def content_from_assistant(self,
                               message: ChatCompletionAssistantMessage) -> str:
        text = ""
        for call in message.tool_calls:
            text += call.id + " was called with arguments : " + str(
                call.function.arguments) + "\n"
        if message.content is None:
            return text
        else:
            return message.content + "\n" + text

    def content_from_tool(self, message: ChatCompletionToolMessage) -> str:
        return message.tool_call_id + " -> " + message.content

    def inject_prompt(self, request: ChatCompletionRequest):
        """ Tested with :
                https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B/discussions/3 """
        if request.tool_choice is not None and request.tools is not None and request.tool_choice == "auto":
            tools_list: [ChatCompletionToolParam] = request.tools
            if len(tools_list):
                text_inject = "The following is a list of external functions that may be called to complete certain tasks:"
                text_inject += "\n["
                for tool in tools_list:
                    if tool.type == "function":
                        json_schema_params = json.dumps(
                            tool.function.parameters, indent=4) if (
                                tool.function.parameters is not None
                                and len(tool.function.parameters)) else None
                        if json_schema_params is not None:
                            text_inject += f"\n  {{\"name\": \"{tool.function.name}\", \"description\": \"{tool.function.description}\", \"arguments\": {json_schema_params}]}},"
                        else:
                            text_inject += f"\n  {{\"name\": \"{tool.function.name}\", \"description\": \"{tool.function.description}\", \"arguments\": null]}},"
                text_inject += "\n]\n"
                text_inject += (
                    f"Whenever the user asks you something, you can either respond directly or invoke a function. "
                    f"The decision to invoke a function is yours, only invoke functions when it makes sense to do so.\n"
                    f"If you have to call at least one function, your message can contain only function calls and nothing else.\n"
                    f"To call a function, the message must start by \"{self.func_call_token()}\" followed by a json like this:\n"
                    f"With arguments:\n"
                    f"  {self.func_call_token()}{{\"call\": \"function_name\", \"arguments\": {{\"arg1\": \"value1\"}}}}.\n"
                    f"Without arguments:\n"
                    f"  {self.func_call_token()}{{\"call\": \"function_name\", \"arguments\": null}}.\n"
                    f"End of functions instructions.\n\n")
                if isinstance(request.messages, str):
                    request.messages = text_inject + request.messages
                elif isinstance(request.messages,
                                List) and len(request.messages) >= 1:
                    request.messages[
                        0].content = text_inject + request.messages[0].content


class PromptCapture:

    def __init__(self):
        self.content: str = ""
        self.maybe_function_call = False
        self.is_function_call = False
        self.prefix_size = 0
        self.calls_list: List[{}] = []

    def reset(self, reset_calls_list=False):
        self.content = ""
        self.maybe_function_call = False
        self.is_function_call = False
        self.prefix_size = 0
        if reset_calls_list:
            self.calls_list = []

    def num_calls(self):
        return len(self.calls_list)

    def make_calls_list(self, prompter: OpenAIToolsPrompter):
        calls_list = self.content.split(prompter.func_call_token())
        for v_call in calls_list:
            if len(v_call):
                try:
                    call_dict = json.loads(v_call)
                    if "call" in call_dict:
                        self.calls_list.append(call_dict)
                except json.decoder.JSONDecodeError:
                    # Simply ignore invalid functions calls...
                    pass

    def validate_call(self, call_id: int, tools_list: [str]) -> int:
        """ Validate function / tool calls by searching name in the tools defined in the request.
            Returns the function id or -1 on failure."""
        if len(self.calls_list) and call_id < len(self.calls_list):
            try:
                return tools_list.index(self.calls_list[call_id]["call"])
            except ValueError:
                pass
        return -1

    def to_ToolCallsMessage(self,
                            call_id: int) -> Union[ToolCallsMessage, None]:
        if len(self.calls_list) and call_id < len(self.calls_list):
            call = self.calls_list[call_id]
            arguments = call["arguments"] if "arguments" in call else None
            function_call = FunctionCall(name=call["call"],
                                         arguments=json.dumps(arguments)
                                         if arguments is not None else "")
            return ToolCallsMessage(id="call_" + call["call"] + "_" +
                                    str(call_id),
                                    type="function",
                                    function=function_call)
        return None

    def to_ToolCallsDelta(self, call_id: int) -> Union[ToolCallsDelta, None]:
        mesg = self.to_ToolCallsMessage(call_id)
        if mesg is not None:
            return ToolCallsDelta(index=call_id,
                                  id=mesg.id,
                                  type=mesg.type,
                                  function=mesg.function)
        return None


class OpenAIServing:

    def __init__(self,
                 engine: AsyncLLMEngine,
                 served_model: str,
                 response_role: str,
                 chat_template=None,
                 openai_tools_prompter: OpenAIToolsPrompter = None):
        self.engine = engine
        self.openai_tools_prompter = openai_tools_prompter
        self.served_model = served_model
        self.chat_template = chat_template
        self.response_role = response_role

        self.max_model_len = 0
        self.tokenizer = None

        try:
            if self.engine.engine_use_ray and asyncio.get_event_loop(
            ) is not None:
                asyncio.create_task(self._post_init())
            else:
                asyncio.run(self._post_init())
        except RuntimeError:
            asyncio.run(self._post_init())

    async def _post_init(self):
        engine_model_config = await self.engine.get_model_config()
        self.max_model_len = engine_model_config.max_model_len

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            engine_model_config.tokenizer,
            tokenizer_mode=engine_model_config.tokenizer_mode,
            trust_remote_code=engine_model_config.trust_remote_code)
        self._load_chat_template(self.chat_template)

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        return ModelList(data=[
            ModelCard(id=self.served_model,
                      root=self.served_model,
                      permission=[ModelPermission()])
        ])

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See  https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI ChatCompletion API.

        NOTE: Currently we do not support the following features:
            - logit_bias (to be supported by vLLM engine)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if request.logit_bias is not None and len(request.logit_bias) > 0:
            # TODO: support logit_bias in vLLM engine.
            return self.create_error_response(
                "logit_bias is not currently supported")

        if self.openai_tools_prompter is not None:
            self.openai_tools_prompter.inject_prompt(request)

            # FIXME : As on dec 2023, the tokenizer only accept "role" and "content" attributes.
            # FIXME : So we manually copy other attributes into "content" when needed.
            for m in request.messages:
                if isinstance(m, ChatCompletionAssistantMessage
                              ) and m.tool_calls is not None:
                    m.content = self.openai_tools_prompter.content_from_assistant(
                        m)
                elif isinstance(m, ChatCompletionToolMessage
                                ) and m.tool_call_id is not None:
                    m.content = self.openai_tools_prompter.content_from_tool(m)

        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt)
        except Exception as e:
            logger.error(
                f"Error in applying chat template from request: {str(e)}")
            return self.create_error_response(str(e))

        # logger.info(prompt)  # print current prompt

        token_ids, error_check_ret = await self._check_length(request,
                                                              prompt=prompt)
        if error_check_ret is not None:
            return error_check_ret

        try:
            spaces_between_special_tokens = request.spaces_between_special_tokens
            sampling_params = SamplingParams(
                n=request.n,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                repetition_penalty=request.repetition_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                min_p=request.min_p,
                stop=request.stop,
                stop_token_ids=request.stop_token_ids,
                max_tokens=request.max_tokens,
                best_of=request.best_of,
                top_k=request.top_k,
                ignore_eos=request.ignore_eos,
                use_beam_search=request.use_beam_search,
                skip_special_tokens=request.skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens,
            )
        except ValueError as e:
            return self.create_error_response(str(e))

        request_id = f"cmpl-{random_uuid()}"
        result_generator = self.engine.generate(prompt, sampling_params,
                                                request_id, token_ids)

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id)
        else:
            return await self.chat_completion_full_generator(
                request, raw_request, result_generator, request_id)

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1].role

    async def chat_completion_stream_generator(
            self, request: ChatCompletionRequest,
            result_generator: AsyncIterator[RequestOutput], request_id: str
    ) -> Union[ErrorResponse, AsyncGenerator[str, None]]:

        model_name = request.model
        created_time = int(time.monotonic())
        chunk_object_type = "chat.completion.chunk"

        # Send first response for each request.n (index) with the role
        role = self.get_chat_request_role(request)
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role=role), finish_reason=None)
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 object=chunk_object_type,
                                                 created=created_time,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if request.echo:
            last_msg_content = ""
            if request.messages and \
                    isinstance(request.messages, list) and \
                    request.messages[-1].content and \
                    request.messages[-1].role == role:
                last_msg_content = request.messages[-1].content
            if last_msg_content:
                for i in range(request.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=last_msg_content),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"

        tools_capture_texts = None
        tools_list = None
        if self.openai_tools_prompter is not None and request.tools is not None:
            tools_capture_texts = [PromptCapture()] * request.n
            tools_list: [str] = [tool.function.name for tool in request.tools]

        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index

                if finish_reason_sent[i]:
                    continue

                current_capture = tools_capture_texts[
                    i] if tools_capture_texts is not None else None

                # Manage tools calling
                if self.openai_tools_prompter is not None and \
                        request.tools is not None and \
                        output.finish_reason is None:
                    if len(current_capture.content) == 0:
                        current_token: str = output.text[len(previous_texts[i]
                                                             ):]
                        if OpenAIToolsPrompter.func_call_token_pre(
                        ) in current_token:
                            start_pos: int = current_token.index(
                                OpenAIToolsPrompter.func_call_token_pre())
                            current_capture.content = current_token[
                                start_pos:]  # With some models the completion may start by a space.
                            current_capture.prefix_size = len(
                                output.text) - len(current_capture.content)
                            current_capture.maybe_function_call = True
                    else:  # Maybe a function call...
                        current_token: str = output.text[
                            len(current_capture.content) +
                            current_capture.prefix_size:]
                        current_capture.content += current_token
                        if len(current_capture.content
                               ) < OpenAIToolsPrompter.func_call_token_size():
                            pass
                        elif not current_capture.is_function_call:
                            if current_capture.content.startswith(
                                    OpenAIToolsPrompter.func_call_token(
                                    )):  # Function call !
                                current_capture.is_function_call = True
                            else:  # This is not a function call...
                                current_capture.reset(False)
                        else:  # Currently extracting the function call
                            if current_capture.content.rfind("}", -6) != -1:
                                c1 = current_capture.content.count("{")
                                c2 = current_capture.content.count("}")
                                if c1 == c2:  # We have the complete call block
                                    previous_texts[i] = output.text
                                    current_capture.make_calls_list(
                                        self.openai_tools_prompter)
                                    current_capture.reset(False)
                            else:
                                pass
                if current_capture is None or (
                        not current_capture.maybe_function_call):
                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        delta_text = output.text[len(previous_texts[i]):]
                        previous_texts[i] = output.text
                        previous_num_tokens[i] = len(output.token_ids)
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(content=delta_text),
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        data = chunk.json(exclude_unset=True,
                                          ensure_ascii=False)
                        yield f"data: {data}\n\n"
                    else:
                        if output.finish_reason == "stop" and (
                                current_capture is not None and
                            (current_capture.num_calls() > 0)):
                            calls_count = current_capture.num_calls()
                            tools_calls_list = []
                            for call_id in range(calls_count):
                                func_id = current_capture.validate_call(
                                    call_id, tools_list)
                                if func_id >= 0:
                                    tools_calls_list.append(
                                        current_capture.to_ToolCallsDelta(
                                            call_id=call_id))
                            choice_data = ChatCompletionResponseStreamChoice(
                                index=i,
                                delta=DeltaMessage(
                                    content=None, tool_calls=tools_calls_list),
                                finish_reason="tool_calls")
                            chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                object=chunk_object_type,
                                created=created_time,
                                choices=[choice_data],
                                model=model_name)
                            chunk.usage = UsageInfo(
                                prompt_tokens=len(res.prompt_token_ids),
                                completion_tokens=len(output.token_ids),
                                total_tokens=len(res.prompt_token_ids) +
                                len(output.token_ids),
                            )
                            data = chunk.json(exclude_unset=True,
                                              exclude_none=True,
                                              ensure_ascii=False)
                            yield f"data: {data}\n\n"
                        else:
                            # Send the finish response for each request.n only once
                            prompt_tokens = len(res.prompt_token_ids)
                            final_usage = UsageInfo(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=previous_num_tokens[i],
                                total_tokens=prompt_tokens +
                                previous_num_tokens[i],
                            )
                            choice_data = ChatCompletionResponseStreamChoice(
                                index=i,
                                delta=[],
                                finish_reason=output.finish_reason)
                            chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                object=chunk_object_type,
                                created=created_time,
                                choices=[choice_data],
                                model=model_name)
                            if final_usage is not None:
                                chunk.usage = final_usage
                            data = chunk.json(exclude_unset=True,
                                              exclude_none=True,
                                              ensure_ascii=False)
                            yield f"data: {data}\n\n"
                        finish_reason_sent[i] = True
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
            self, request: ChatCompletionRequest, raw_request: Request,
            result_generator: AsyncIterator[RequestOutput],
            request_id: str) -> Union[ErrorResponse, ChatCompletionResponse]:
        model_name = request.model
        created_time = int(time.monotonic())

        final_res: RequestOutput = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices = []
        role = self.get_chat_request_role(request)
        for output in final_res.outputs:

            # logger.info(output.text)
            tools_calls_validation = False

            # Manage tools calling
            if self.openai_tools_prompter is not None and \
                    request.tools is not None:
                tools_list: [str] = [
                    tool.function.name for tool in request.tools
                ]
                current_capture = PromptCapture()
                # current_capture.content = output.text.lstrip(" ")  # With some models the completion may start by a space.

                start_pos = 0
                while True:
                    pos = output.text.find(
                        OpenAIToolsPrompter.func_call_token(), start_pos, -1)
                    if pos < 0:
                        break
                    start_bloc = output.text.find("{", pos, -1)
                    if start_bloc < 0:
                        break
                    if (start_bloc -
                        (pos +
                         OpenAIToolsPrompter.func_call_token_size())) > 1:
                        break
                    count = 1
                    bloc_end = start_bloc + 1
                    for it_ch in range(start_bloc + 1, len(output.text), 1):
                        ch = output.text[it_ch]
                        bloc_end += 1
                        if ch == "{":
                            count += 1
                        elif ch == "}":
                            count -= 1
                        if count == 0:  # We have the complete call block
                            current_capture.content = output.text[
                                start_bloc:bloc_end]
                            current_capture.make_calls_list(
                                self.openai_tools_prompter)
                            current_capture.reset(False)
                            break
                    start_pos = bloc_end + 1

                if current_capture.num_calls() > 0:
                    tools_calls_validation = True
                    calls_count = current_capture.num_calls()
                    tools_calls_list = []
                    for call_id in range(calls_count):
                        func_id = current_capture.validate_call(
                            call_id, tools_list)
                        if func_id >= 0:
                            tools_calls_list.append(
                                current_capture.to_ToolCallsMessage(
                                    call_id=call_id))
                    message = ChatMessage(role=role,
                                          content=None,
                                          tool_calls=tools_calls_list)
                    choice_data = ChatCompletionResponseChoice(
                        index=output.index,
                        message=message,
                        finish_reason="tool_calls")
                    choices.append(choice_data)
            if not tools_calls_validation:
                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=ChatMessage(role=role, content=output.text),
                    finish_reason=output.finish_reason,
                )
                choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if request.messages and \
                    isinstance(request.messages, list) and \
                    request.messages[-1].content and \
                    request.messages[-1].role == role:
                last_msg_content = request.messages[-1].content

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )
        return response

    async def create_completion(self, request: CompletionRequest,
                                raw_request: Request):
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following features:
            - suffix (the language models we currently support do not support
              suffix)
            - logit_bias (to be supported by vLLM engine)
        """

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # OpenAI API supports echoing the prompt when max_tokens is 0.
        echo_without_generation = request.echo and request.max_tokens == 0

        if request.suffix is not None:
            # The language models we currently support do not support suffix.
            return self.create_error_response(
                "suffix is not currently supported")

        if request.logit_bias is not None and len(request.logit_bias) > 0:
            # TODO: support logit_bias in vLLM engine.
            return self.create_error_response(
                "logit_bias is not currently supported")

        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"

        use_token_ids = False
        if isinstance(request.prompt, list):
            if len(request.prompt) == 0:
                return self.create_error_response(
                    "please provide at least one prompt")
            first_element = request.prompt[0]
            if isinstance(first_element, int):
                use_token_ids = True
                prompt = request.prompt
            elif isinstance(first_element, (str, list)):
                # TODO: handles multiple prompt case in list[list[int]]
                if len(request.prompt) > 1:
                    return self.create_error_response(
                        "multiple prompts in a batch is not currently supported"
                    )
                use_token_ids = not isinstance(first_element, str)
                prompt = request.prompt[0]
        else:
            prompt = request.prompt

        if use_token_ids:
            _, error_check_ret = await self._check_length(request,
                                                          prompt_ids=prompt)
        else:
            token_ids, error_check_ret = await self._check_length(
                request, prompt=prompt)
        if error_check_ret is not None:
            return error_check_ret

        created_time = int(time.monotonic())
        try:
            spaces_between_special_tokens = request.spaces_between_special_tokens
            sampling_params = SamplingParams(
                n=request.n,
                best_of=request.best_of,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                repetition_penalty=request.repetition_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                stop=request.stop,
                stop_token_ids=request.stop_token_ids,
                ignore_eos=request.ignore_eos,
                max_tokens=request.max_tokens
                if not echo_without_generation else 1,
                logprobs=request.logprobs,
                use_beam_search=request.use_beam_search,
                prompt_logprobs=request.logprobs if request.echo else None,
                skip_special_tokens=request.skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens,
            )
        except ValueError as e:
            return self.create_error_response(str(e))

        if use_token_ids:
            result_generator = self.engine.generate(None,
                                                    sampling_params,
                                                    request_id,
                                                    prompt_token_ids=prompt)
        else:
            result_generator = self.engine.generate(prompt, sampling_params,
                                                    request_id, token_ids)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. In addition, we do not stream the results when use beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        def create_stream_response_json(
            index: int,
            text: str,
            logprobs: Optional[LogProbs] = None,
            finish_reason: Optional[str] = None,
            usage: Optional[UsageInfo] = None,
        ) -> str:
            choice_data = CompletionResponseStreamChoice(
                index=index,
                text=text,
                logprobs=logprobs,
                finish_reason=finish_reason,
            )
            response = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[choice_data],
            )
            if usage is not None:
                response.usage = usage
            response_json = response.json(exclude_unset=True,
                                          ensure_ascii=False)

            return response_json

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            previous_texts = [""] * request.n
            previous_num_tokens = [0] * request.n
            has_echoed = [False] * request.n
            async for res in result_generator:
                res: RequestOutput
                for output in res.outputs:
                    i = output.index
                    delta_text = output.text[len(previous_texts[i]):]
                    token_ids = output.token_ids[previous_num_tokens[i]:]
                    if request.logprobs is not None:
                        top_logprobs = output.logprobs[previous_num_tokens[i]:]
                    else:
                        top_logprobs = None
                    offsets = len(previous_texts[i])
                    if request.echo and not has_echoed[i]:
                        if not echo_without_generation:
                            delta_text = res.prompt + delta_text
                            token_ids = res.prompt_token_ids + token_ids
                            if top_logprobs:
                                top_logprobs = res.prompt_logprobs + top_logprobs
                        else:  # only just return the prompt
                            delta_text = res.prompt
                            token_ids = res.prompt_token_ids
                            if top_logprobs:
                                top_logprobs = res.prompt_logprobs
                        has_echoed[i] = True
                    if request.logprobs is not None:
                        logprobs = self._create_logprobs(
                            token_ids=token_ids,
                            top_logprobs=top_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            initial_text_offset=offsets,
                        )
                    else:
                        logprobs = None
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    finish_reason = output.finish_reason
                    response_json = create_stream_response_json(
                        index=i,
                        text=delta_text,
                        logprobs=logprobs,
                        finish_reason=finish_reason,
                    )
                    yield f"data: {response_json}\n\n"
                    if output.finish_reason is not None:
                        logprobs = (LogProbs()
                                    if request.logprobs is not None else None)
                        prompt_tokens = len(res.prompt_token_ids)
                        completion_tokens = len(output.token_ids)
                        final_usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        response_json = create_stream_response_json(
                            index=i,
                            text="",
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                            usage=final_usage,
                        )
                        yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        # Streaming response
        if stream:
            return completion_stream_generator()

        # Non-streaming response
        final_res: RequestOutput = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None
        choices = []
        prompt_token_ids = final_res.prompt_token_ids
        prompt_logprobs = final_res.prompt_logprobs
        prompt_text = final_res.prompt
        for output in final_res.outputs:
            if request.logprobs is not None:
                if not echo_without_generation:
                    token_ids = output.token_ids
                    top_logprobs = output.logprobs
                    if request.echo:
                        token_ids = prompt_token_ids + token_ids
                        top_logprobs = prompt_logprobs + top_logprobs
                else:
                    token_ids = prompt_token_ids
                    top_logprobs = prompt_logprobs
                logprobs = self._create_logprobs(
                    token_ids=token_ids,
                    top_logprobs=top_logprobs,
                    num_output_top_logprobs=request.logprobs,
                )
            else:
                logprobs = None
            if not echo_without_generation:
                output_text = output.text
                if request.echo:
                    output_text = prompt_text + output_text
            else:
                output_text = prompt_text
            choice_data = CompletionResponseChoice(
                index=output.index,
                text=output_text,
                logprobs=logprobs,
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        if request.stream:
            # When user requests streaming but we don't stream, we still need to
            # return a streaming response with a single event.
            response_json = response.json(ensure_ascii=False)

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response

    def _create_logprobs(
        self,
        token_ids: List[int],
        top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None,
        num_output_top_logprobs: Optional[int] = None,
        initial_text_offset: int = 0,
    ) -> LogProbs:
        """Create OpenAI-style logprobs."""
        logprobs = LogProbs()
        last_token_len = 0
        if num_output_top_logprobs:
            logprobs.top_logprobs = []
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is not None:
                token_logprob = step_top_logprobs[token_id]
            else:
                token_logprob = None
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            logprobs.tokens.append(token)
            logprobs.token_logprobs.append(token_logprob)
            if len(logprobs.text_offset) == 0:
                logprobs.text_offset.append(initial_text_offset)
            else:
                logprobs.text_offset.append(logprobs.text_offset[-1] +
                                            last_token_len)
            last_token_len = len(token)

            if num_output_top_logprobs:
                logprobs.top_logprobs.append({
                    self.tokenizer.convert_ids_to_tokens(i): p
                    for i, p in step_top_logprobs.items()
                } if step_top_logprobs else None)
        return logprobs

    def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message,
                             type=err_type,
                             code=status_code.value)

    async def create_error_generator(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
    ) -> AsyncGenerator[str, None]:
        yield ErrorResponse(message=message,
                            type=err_type,
                            code=status_code.value).json(ensure_ascii=False)

    async def _check_model(self, request) -> Optional[ErrorResponse]:
        if request.model == self.served_model:
            return
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)

    async def _check_length(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest],
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None
    ) -> Tuple[List[int], Optional[ErrorResponse]]:
        assert (not (prompt is None and prompt_ids is None)
                and not (prompt is not None and prompt_ids is not None)
                ), "Either prompt or prompt_ids should be provided."
        input_ids = prompt_ids if prompt_ids is not None else self.tokenizer(
            prompt).input_ids
        token_num = len(input_ids)

        if request.max_tokens is None:
            request.max_tokens = self.max_model_len - token_num
        if token_num + request.max_tokens > self.max_model_len:
            return input_ids, self.create_error_response(
                f"This model's maximum context length is {self.max_model_len} tokens. "
                f"However, you requested {request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.", )
        else:
            return input_ids, None

    def _load_chat_template(self, chat_template):
        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    self.tokenizer.chat_template = f.read()
            except OSError:
                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                self.tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logger.info(
                f"Using supplied chat template:\n{self.tokenizer.chat_template}"
            )
        elif self.tokenizer.chat_template is not None:
            logger.info(
                f"Using default chat template:\n{self.tokenizer.chat_template}"
            )
        else:
            logger.warning(
                "No chat template provided. Chat API will not work.")
