import time
import codecs
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Union
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    UsageInfo)
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 served_model: str,
                 response_role: str,
                 chat_template=None):
        super().__init__(engine=engine, served_model=served_model)
        self.response_role = response_role
        self._load_chat_template(chat_template)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See  https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI ChatCompletion API.

        NOTE: Currently we do not support the following features:
            - function_call (Users should implement this by themselves)
            - logit_bias (to be supported by vLLM engine)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if request.logit_bias is not None and len(request.logit_bias) > 0:
            # TODO: support logit_bias in vLLM engine.
            return self.create_error_response(
                "logit_bias is not currently supported")

        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt)
        except Exception as e:
            logger.error(
                f"Error in applying chat template from request: {str(e)}")
            return self.create_error_response(str(e))

        request_id = f"cmpl-{random_uuid()}"
        try:
            token_ids = self._validate_prompt_and_tokenize(request,
                                                           prompt=prompt)
            sampling_params = request.to_sampling_params()
        except ValueError as e:
            return self.create_error_response(str(e))

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
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]
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
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

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

                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                if output.finish_reason is None:
                    # Send token-by-token response for each request.n
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
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                else:
                    # Send the finish response for each request.n only once
                    prompt_tokens = len(res.prompt_token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=previous_num_tokens[i],
                        total_tokens=prompt_tokens + previous_num_tokens[i],
                    )
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=delta_text),
                        finish_reason=output.finish_reason)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    if final_usage is not None:
                        chunk.usage = final_usage
                    data = chunk.model_dump_json(exclude_unset=True,
                                                 exclude_none=True)
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
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
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
