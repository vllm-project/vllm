import time
import codecs
import asyncio
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Optional, List, Union
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    VllmToolsTemplate, ChatCompletionAssistantMessage,
    ChatCompletionToolMessage, ChatCompletionNamedToolChoiceParam,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    UsageInfo)
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing, LoRA
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
from vllm.entrypoints.openai.tools import OpenAIToolsPrompter, ChatPromptCapture

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 served_model: str,
                 response_role: str,
                 lora_modules: Optional[List[LoRA]] = None,
                 chat_template=None,
                 openai_tools_prompter: OpenAIToolsPrompter = None,
                 privileged: bool = False):
        super().__init__(engine=engine,
                         served_model=served_model,
                         lora_modules=lora_modules,
                         privileged=privileged)
        self.response_role = response_role
        self.default_tools_template = VllmToolsTemplate()
        self.openai_tools_prompter = openai_tools_prompter

        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        if event_loop is not None and event_loop.is_running(
        ):  # If the current is instanced by Ray Serve, there is already a running event loop
            event_loop.create_task(self._load_chat_template(chat_template))
        else:  # When using single vLLM without engine_use_ray
            asyncio.run(self._load_chat_template(chat_template))

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See  https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI ChatCompletion API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if self.openai_tools_prompter is not None:
            if isinstance(request.tool_params, VllmToolsTemplate):
                if len(request.tool_params.call_token_start) == 0:
                    raise ValueError(
                        "Error, the tool_params.call_token_start can't be empty !"
                    )
            else:
                # No need to allocate it for each requests, if this param is not set, we use the default value.
                request.tool_params = self.default_tools_template
            self.openai_tools_prompter.inject_prompt(request)

            # FIXME : As on dec 2023, the tokenizer only accept "role" and "content" attributes.
            # FIXME : So we manually copy other attributes into "content" when needed.
            for m in request.messages:
                if isinstance(m, ChatCompletionAssistantMessage
                              ) and m.tool_calls is not None:
                    m.content = self.openai_tools_prompter.content_from_assistant(
                        m, request.tool_params)
                elif isinstance(m, ChatCompletionToolMessage
                                ) and m.tool_call_id is not None:
                    m.content = self.openai_tools_prompter.content_from_tool(
                        m, request.tool_params)

        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt)
        except Exception as e:
            logger.error(
                f"Error in applying chat template from request: {str(e)}")
            return self.create_error_response(str(e))

        if self.privileged:  # ease the templates development
            logger.info("\n######## Development infos (dev-mode) ########")
            logger.info("API tools status: %s" %
                        str(self.openai_tools_prompter is not None))
            logger.info("- Request:\n%s" % str(request.dict()))
            logger.info("")
            logger.info("- Prompt:\n%s" % str(prompt))
            logger.info("##############################################")

        request_id = f"cmpl-{random_uuid()}"
        try:
            token_ids = self._validate_prompt_and_tokenize(request,
                                                           prompt=prompt)
            sampling_params = request.to_sampling_params()
            lora_request = self._maybe_get_lora(request)
            guided_decode_logits_processor = (
                await get_guided_decoding_logits_processor(
                    request, self.engine.get_tokenizer()))
            if guided_decode_logits_processor:
                if sampling_params.logits_processors is None:
                    sampling_params.logits_processors = []
                sampling_params.logits_processors.append(
                    guided_decode_logits_processor)
        except ValueError as e:
            return self.create_error_response(str(e))

        result_generator = self.engine.generate(prompt, sampling_params,
                                                request_id, token_ids,
                                                lora_request)
        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id)
        else:
            try:
                return await self.chat_completion_full_generator(
                    request, raw_request, result_generator, request_id)
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
            self, request: ChatCompletionRequest,
            result_generator: AsyncIterator[RequestOutput], request_id: str
    ) -> Union[ErrorResponse, AsyncGenerator[str, None]]:

        model_name = request.model
        created_time = int(time.monotonic())
        chunk_object_type = "chat.completion.chunk"
        first_iteration = True

        if isinstance(
                request.tool_choice,
                ChatCompletionNamedToolChoiceParam):  # Guided function call
            tools_capture_texts = [
                ChatPromptCapture(self.openai_tools_prompter,
                                  request.tool_params)
                for i in range(request.n)
            ]
            is_tools_guided_generation = True
        else:
            is_tools_guided_generation = False
            if self.openai_tools_prompter is not None and (
                    isinstance(request.tool_choice, str)
                    and request.tool_choice == "auto"):
                tools_capture_texts = [
                    ChatPromptCapture(self.openai_tools_prompter,
                                      request.tool_params)
                    for i in range(request.n)
                ]
            else:
                tools_capture_texts = None

        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        try:
            async for res in result_generator:
                res: RequestOutput
                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with the role
                    role = self.get_chat_request_role(request)
                    for i in range(request.n):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role=role),
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the last message
                    if request.echo:
                        last_msg_content = ""
                        if request.messages and isinstance(
                                request.messages,
                                list) and request.messages[-1].get(
                                    "content") and request.messages[-1].get(
                                        "role") == role:
                            last_msg_content = request.messages[-1]["content"]

                        if last_msg_content:
                            for i in range(request.n):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(
                                        content=last_msg_content),
                                    finish_reason=None)
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    logprobs=None,
                                    model=model_name)
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    current_capture = tools_capture_texts[
                        i] if tools_capture_texts is not None else None

                    if current_capture is not None and current_capture.after_new_function_call:
                        current_capture.after_new_function_call = False
                        # If the last token is a new line char right after a function call, we ignore it.
                        # Otherwise, each function call creates a line break in the content part of the response.
                        if output.text[len(previous_texts[i]):] == "\n":
                            previous_texts[i] = output.text
                            continue

                    delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                    top_logprobs = output.logprobs[
                        previous_num_tokens[i]:] if output.logprobs else None

                    if request.logprobs:
                        logprobs = self._create_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=top_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            initial_text_offset=len(previous_texts[i]),
                        )
                    else:
                        logprobs = None

                    if is_tools_guided_generation:  # Manage tools calling when request.tool_choice set a function
                        if len(current_capture.content) == 0:
                            current_capture.startNamedFunction(
                                request.tool_choice)
                        current_token: str = output.text[len(previous_texts[i]
                                                             ):]
                        if len(current_token):
                            current_capture.content += current_token
                            if current_capture.checkBracketsFunctionCall(
                                    request.tool_params
                            ):  # We have the complete call block
                                previous_texts[i] = output.text
                                current_capture.closeNamedFunction()
                                current_capture.make_calls_list()
                                current_capture.reset(False)
                                current_capture.after_new_function_call = True
                    else:  # Manage tools calling when request.tool_choice is "auto"
                        if (self.openai_tools_prompter is not None) and \
                                (current_capture is not None) and \
                                (request.tools is not None) and \
                                (output.finish_reason is None):
                            if len(current_capture.content) == 0:
                                current_token: str = output.text[
                                    len(previous_texts[i]):]
                                if current_capture.func_call_token_pre(
                                ) in current_token:
                                    start_pos: int = current_token.index(
                                        current_capture.func_call_token_pre())
                                    current_capture.content = current_token[
                                        start_pos:]  # With some models the completion may start by a space.
                                    current_capture.prefix_size = len(
                                        output.text) - len(
                                            current_capture.content)
                                    current_capture.maybe_function_call = True
                            else:  # Maybe a function call...
                                current_token: str = output.text[
                                    len(current_capture.content) +
                                    current_capture.prefix_size:]
                                current_capture.content += current_token
                                if len(
                                        current_capture.content
                                ) < current_capture.func_call_token_size():
                                    pass
                                elif not current_capture.is_function_call:
                                    if current_capture.content.startswith(
                                            current_capture.func_call_token(
                                            )):  # Function call !
                                        current_capture.is_function_call = True
                                    else:  # This is not a function call...
                                        current_capture.reset(False)
                                else:  # Currently extracting the function call
                                    if current_capture.checkBracketsFunctionCall(
                                            request.tool_params
                                    ):  # We have the complete call block
                                        previous_texts[i] = output.text
                                        current_capture.make_calls_list()
                                        current_capture.reset(False)
                                        current_capture.after_new_function_call = True
                                    else:
                                        pass

                    if current_capture is None or (
                            isinstance(current_capture, ChatPromptCapture)
                            and not current_capture.maybe_function_call):
                        delta_text = output.text[len(previous_texts[i]):]
                        previous_texts[i] = output.text
                        previous_num_tokens[i] = len(output.token_ids)
                        if output.finish_reason is None:
                            if len(delta_text) > 0:
                                # Send token-by-token response for each request.n
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=delta_text),
                                    logprobs=logprobs,
                                    finish_reason=None)
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name)
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                        else:
                            if output.finish_reason == "stop" and (
                                    isinstance(current_capture,
                                               ChatPromptCapture) and
                                (current_capture.num_calls() > 0)):
                                tools_calls_list = current_capture.to_ChoiceDeltaToolCallList(
                                )

                                if self.privileged:
                                    for t in tools_calls_list:
                                        logger.warning(
                                            "Calling tools: %s" %
                                            str(t.model_dump_json()))

                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(
                                        content=None,
                                        tool_calls=tools_calls_list),
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
                                data = chunk.model_dump_json(
                                    exclude_unset=True, exclude_none=True)
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
                                    delta=DeltaMessage(content=delta_text),
                                    logprobs=logprobs,
                                    finish_reason=output.finish_reason)
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name)
                                if final_usage is not None:
                                    chunk.usage = final_usage
                                data = chunk.model_dump_json(
                                    exclude_unset=True, exclude_none=True)
                                yield f"data: {data}\n\n"
                                finish_reason_sent[i] = True
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
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
            token_ids = output.token_ids
            top_logprobs = output.logprobs
            tools_calls_validation = False

            if request.logprobs:
                logprobs = self._create_logprobs(
                    token_ids=token_ids,
                    top_logprobs=top_logprobs,
                    num_output_top_logprobs=request.logprobs,
                )
            else:
                logprobs = None

            # Manage tools calling
            if self.openai_tools_prompter is not None and \
                    request.tools is not None:
                current_capture = ChatPromptCapture(self.openai_tools_prompter,
                                                    request.tool_params)

                if isinstance(request.tool_choice,
                              ChatCompletionNamedToolChoiceParam
                              ):  # Guided function call
                    current_capture.startNamedFunction(request.tool_choice)
                    current_capture.content += output.text
                    current_capture.closeNamedFunction()
                    current_capture.make_calls_list()
                    current_capture.reset(False)
                else:
                    start_pos = 0
                    while True:
                        pos = output.text.find(
                            current_capture.func_call_token(), start_pos, -1)
                        if pos < 0:
                            break
                        start_bloc = output.text.find("{", pos, -1)
                        if start_bloc < 0:
                            break
                        if (start_bloc -
                            (pos +
                             current_capture.func_call_token_size())) > 1:
                            break
                        count = 1
                        bloc_end = start_bloc + 1
                        for it_ch in range(start_bloc + 1, len(output.text),
                                           1):
                            ch = output.text[it_ch]
                            bloc_end += 1
                            if ch == "{":
                                count += 1
                            elif ch == "}":
                                count -= 1
                            if count == 0:  # We have the complete call block
                                current_capture.content = output.text[
                                    start_bloc:bloc_end]
                                current_capture.make_calls_list()
                                current_capture.reset(False)
                                break
                        start_pos = bloc_end + 1

                if current_capture.num_calls() > 0:
                    tools_calls_validation = True
                    tools_calls_list = current_capture.to_ChatCompletionMessageToolCallList(
                    )
                    message = ChatMessage(role=role,
                                          content=None,
                                          tool_calls=tools_calls_list)
                    choice_data = ChatCompletionResponseChoice(
                        index=output.index,
                        message=message,
                        logprobs=logprobs,
                        finish_reason="tool_calls")
                    choices.append(choice_data)
            if not tools_calls_validation:
                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=ChatMessage(role=role, content=output.text),
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                )
                choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]

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

    async def _load_chat_template(self, chat_template):
        while self.tokenizer is None:
            logger.info("Waiting for the tokenizer initialization...")
            await asyncio.sleep(1.00)

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
