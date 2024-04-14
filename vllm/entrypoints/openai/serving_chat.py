import codecs
import time
from typing import (AsyncGenerator, AsyncIterator, Awaitable, Iterable, List,
                    Optional, Tuple, TypedDict, Union, final)

import numpy as np
import torch
from fastapi import Request
from openai.types.chat import (ChatCompletionContentPartParam,
                               ChatCompletionRole)

from vllm.config import VisionLanguageConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    UsageInfo)
from vllm.entrypoints.openai.serving_engine import (LoRAModulePath,
                                                    OpenAIServing)
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.outputs import RequestOutput
from vllm.sequence import MultiModalData
from vllm.utils import get_image_async, random_uuid

logger = init_logger(__name__)


async def get_and_parse_image(image_url: str,
                              config: VisionLanguageConfig) -> MultiModalData:

    if len(config.image_input_shape) == 3:
        raise ValueError(
            "The model is configured to accept image features rather than "
            "pixel values, and thus does not support image inputs")

    batch_size, num_channels, height, width = config.image_input_shape

    if num_channels == 1:
        image_format = "L"
    elif num_channels == 3:
        image_format = "RGB"
    elif num_channels == 4:
        image_format = "RGBA"
    else:
        msg = f"Unsupported number of channels ({num_channels})"
        raise NotImplementedError(msg)

    with await get_image_async(image_url) as image:
        image = image.convert(image_format).resize((height, width))
        image_arr = np.array(image, copy=True)

    # Passed to the image processor which is loaded from HuggingFace
    image_tensor = torch.as_tensor(image_arr) \
        .view(batch_size, height, width, num_channels) \
        .permute((0, 3, 1, 2))  # NCHW

    return MultiModalData(type=MultiModalData.Type.IMAGE, data=image_tensor)


@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: str


class OpenAIServingChat(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 served_model: str,
                 response_role: str,
                 lora_modules: Optional[List[LoRAModulePath]] = None,
                 chat_template=None):
        super().__init__(engine=engine,
                         served_model=served_model,
                         lora_modules=lora_modules)
        self.response_role = response_role
        self._load_chat_template(chat_template)

    def _parse_chat_message_image_input(
        self,
        role: ChatCompletionRole,
        content: Iterable[ChatCompletionContentPartParam],
    ) -> Tuple[List[ConversationMessage], List[Awaitable[MultiModalData]]]:
        """Parse image input defined by OpenAI Chat Completions API."""
        config = getattr(self.engine.engine, "vision_language_config", None)
        if not isinstance(config, VisionLanguageConfig):
            raise ValueError("GPT-4 with Vision API is only supported for "
                             "vision language models.")

        tokenizer = self.tokenizer
        assert tokenizer is not None

        texts: List[str] = []
        image_futures: List[Awaitable[MultiModalData]] = []

        for i, part in enumerate(content):
            if part["type"] == "text":
                text = part["text"]

                texts.append(text)
            elif part["type"] == "image_url":
                image_url = part["image_url"]
                if image_url.get("detail", "auto") != "auto":
                    logger.info("content[%s].image_url.detail is ignored", i)

                text = config.image_openai.value.get_image_token_text(
                    config, tokenizer, image_idx=len(image_futures))
                image_future = get_and_parse_image(image_url["url"], config)

                texts.append(text)
                image_futures.append(image_future)
            else:
                raise NotImplementedError(f"Unknown part type: {part['type']}")

        messages = [ConversationMessage(role=role, content="\n".join(texts))]
        data_futures = image_futures

        return messages, data_futures

    def _parse_chat_message_content(
        self,
        role: ChatCompletionRole,
        content: Optional[Union[str,
                                Iterable[ChatCompletionContentPartParam]]],
    ) -> Tuple[List[ConversationMessage], List[Awaitable[MultiModalData]]]:
        if content is None:
            return [], []
        if isinstance(content, str):
            return [ConversationMessage(role=role, content=content)], []

        return self._parse_chat_message_image_input(role, content)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        ChatCompletion API.

        NOTE: Currently we do not support the following feature:
            - function_call (Users should implement this by themselves)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        try:
            conversation: List[ConversationMessage] = []
            multi_modal_futures: List[Awaitable[MultiModalData]] = []

            for m in request.messages:
                messages, futures = self._parse_chat_message_content(
                    m["role"], m["content"])

                conversation.extend(messages)
                multi_modal_futures.extend(futures)

            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
        except Exception as e:
            logger.error(
                f"Error in applying chat template from request: {str(e)}")
            return self.create_error_response(str(e))

        try:
            if len(multi_modal_futures) == 0:
                multi_modal_data = None
            elif len(multi_modal_futures) == 1:
                multi_modal_data = await multi_modal_futures[0]
            else:
                # multi_modal_datas = await asyncio.gather(*multi_modal_futures)
                raise NotImplementedError("Multiple image input not supported")
        except Exception as e:
            logger.error(f"Error in loading multi-modal data: {str(e)}")
            return self.create_error_response(str(e))

        request_id = f"cmpl-{random_uuid()}"
        try:
            sampling_params = request.to_sampling_params()
            lora_request = self._maybe_get_lora(request)
            guided_decode_logits_processor = (
                await get_guided_decoding_logits_processor(
                    request, await self.engine.get_tokenizer()))
            if guided_decode_logits_processor:
                if sampling_params.logits_processors is None:
                    sampling_params.logits_processors = []
                sampling_params.logits_processors.append(
                    guided_decode_logits_processor)

            prompt_ids, prompt_text = self._tokenize_prompt_input(
                request,
                prompt,
                truncate_prompt_tokens=sampling_params.truncate_prompt_tokens,
            )

            result_generator = self.engine.generate(
                prompt_text,
                sampling_params,
                request_id,
                prompt_ids,
                lora_request=lora_request,
                multi_modal_data=multi_modal_data,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                conversation,
                result_generator,
                request_id,
            )
        else:
            try:
                return await self.chat_completion_full_generator(
                    request,
                    conversation,
                    raw_request,
                    result_generator,
                    request_id,
                )
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        conversation: List[ConversationMessage],
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        model_name = request.model
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_texts = [""] * num_choices
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices

        try:
            async for res in result_generator:
                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)
                    for i in range(num_choices):
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

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content = ""
                        if (conversation and conversation[-1]["content"]
                                and conversation[-1]["role"] == role):
                            last_msg_content = conversation[-1]["content"]

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        logprobs=None,
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name)
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
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

                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    if output.finish_reason is None:
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
                        data = chunk.model_dump_json(exclude_unset=True)
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
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason)
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
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        conversation: List[ConversationMessage],
        raw_request: Request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
    ) -> Union[ErrorResponse, ChatCompletionResponse]:
        model_name = request.model
        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices: List[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            top_logprobs = output.logprobs

            if request.logprobs:
                logprobs = self._create_logprobs(
                    token_ids=token_ids,
                    top_logprobs=top_logprobs,
                    num_output_top_logprobs=request.logprobs,
                )
            else:
                logprobs = None

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
                logprobs=logprobs,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if (conversation and conversation[-1]["content"]
                    and conversation[-1]["role"] == role):
                last_msg_content = conversation[-1]["content"]

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

    def _load_chat_template(self, chat_template: str):
        tokenizer = self.tokenizer
        assert tokenizer is not None

        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    tokenizer.chat_template = f.read()
            except OSError:
                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logger.info(
                f"Using supplied chat template:\n{tokenizer.chat_template}")
        elif tokenizer.chat_template is not None:
            logger.info(
                f"Using default chat template:\n{tokenizer.chat_template}")
        else:
            logger.warning(
                "No chat template provided. Chat API will not work.")
