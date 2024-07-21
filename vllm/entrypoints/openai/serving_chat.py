import time
from typing import (AsyncGenerator, AsyncIterator, Awaitable, Dict, List,
                    Optional)
from typing import Sequence as GenericSequence
from typing import Union

from fastapi import Request
from transformers import PreTrainedTokenizer

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.chat_utils import (ConversationMessage,
                                         load_chat_template,
                                         parse_chat_message_content)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionLogProb, ChatCompletionLogProbs,
    ChatCompletionLogProbsContent, ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    FunctionCall, ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (LoRAModulePath,
                                                    OpenAIServing)
from vllm.inputs import PromptInputs
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.multimodal import MultiModalDataDict
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob
from vllm.tracing import (contains_trace_headers, extract_trace_headers,
                          log_tracing_disabled_warning)
from vllm.utils import random_uuid

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 model_config: ModelConfig,
                 served_model_names: List[str],
                 response_role: str,
                 lora_modules: Optional[List[LoRAModulePath]] = None,
                 chat_template: Optional[str] = None):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=lora_modules)

        self.response_role = response_role

        # If this is None we use the tokenizer's default chat template
        self.chat_template = load_chat_template(chat_template)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None
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
            _, lora_request = self._maybe_get_adapter(request)
            tokenizer = await self.engine.get_tokenizer(lora_request)

            conversation: List[ConversationMessage] = []
            mm_futures: List[Awaitable[MultiModalDataDict]] = []

            for msg in request.messages:
                chat_parsed_result = parse_chat_message_content(
                    msg, self.model_config, tokenizer)

                conversation.extend(chat_parsed_result.messages)
                mm_futures.extend(chat_parsed_result.mm_futures)

            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]

            prompt = tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template or self.chat_template,
                **(request.chat_template_kwargs or {}),
            )
        except Exception as e:
            logger.error("Error in applying chat template from request: %s", e)
            return self.create_error_response(str(e))

        mm_data: Optional[MultiModalDataDict] = None
        try:
            if len(mm_futures):
                # since we support only single mm data currently
                assert len(
                    mm_futures
                ) == 1, "Multiple 'image_url' input is currently not supported."
                mm_data = await mm_futures[0]
        except Exception as e:
            logger.error("Error in loading multi-modal data: %s", e)
            return self.create_error_response(str(e))

        request_id = f"cmpl-{random_uuid()}"
        try:
            # Tokenize/detokenize depending on prompt format (string/token list)
            prompt_ids, prompt_text = await self._validate_prompt_and_tokenize(
                request,
                tokenizer,
                prompt=prompt,
                add_special_tokens=request.add_special_tokens)
            sampling_params = request.to_sampling_params()
            decoding_config = await self.engine.get_decoding_config()
            guided_decoding_backend = request.guided_decoding_backend \
                or decoding_config.guided_decoding_backend
            guided_decode_logits_processor = (
                await
                get_guided_decoding_logits_processor(guided_decoding_backend,
                                                     request, tokenizer))
            if guided_decode_logits_processor:
                if sampling_params.logits_processors is None:
                    sampling_params.logits_processors = []
                sampling_params.logits_processors.append(
                    guided_decode_logits_processor)
        except ValueError as e:
            return self.create_error_response(str(e))

        inputs: PromptInputs = {
            "prompt": prompt_text,
            "prompt_token_ids": prompt_ids,
        }
        if mm_data:
            inputs["multi_modal_data"] = mm_data

        is_tracing_enabled = await self.engine.is_tracing_enabled()
        trace_headers = None
        if is_tracing_enabled and raw_request:
            trace_headers = extract_trace_headers(raw_request.headers)
        if not is_tracing_enabled and raw_request and contains_trace_headers(
                raw_request.headers):
            log_tracing_disabled_warning()

        result_generator = self.engine.generate(
            inputs,
            sampling_params,
            request_id,
            lora_request,
            trace_headers=trace_headers,
        )
        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, conversation, tokenizer)
        else:
            try:
                return await self.chat_completion_full_generator(
                    request, raw_request, result_generator, request_id,
                    conversation, tokenizer)
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
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: List[ConversationMessage],
        tokenizer: PreTrainedTokenizer,
    ) -> AsyncGenerator[str, None]:
        model_name = self.served_model_names[0]
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        assert request.n is not None
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        try:
            async for res in result_generator:
                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
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
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content = ""
                        if conversation and conversation[-1].get(
                                "content") and conversation[-1].get(
                                    "role") == role:
                            last_msg_content = conversation[-1]["content"]

                        if last_msg_content:
                            for i in range(request.n):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    logprobs=None,
                                    model=model_name)
                                if (request.stream_options and
                                        request.stream_options.include_usage):
                                    chunk.usage = None
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                    out_logprobs = output.logprobs[
                        previous_num_tokens[i]:] if output.logprobs else None

                    if request.logprobs and request.top_logprobs is not None:
                        assert out_logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_chat_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)

                    if request.tool_choice and type(
                            request.tool_choice
                    ) is ChatCompletionNamedToolChoiceParam:
                        delta_message = DeltaMessage(tool_calls=[
                            ToolCall(function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=delta_text))
                        ])
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n

                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                    else:
                        # Send the finish response for each request.n only once
                        prompt_tokens = len(res.prompt_token_ids)
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)
                        if (request.stream_options
                                and request.stream_options.include_usage):
                            chunk.usage = None
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                        finish_reason_sent[i] = True

            if (request.stream_options
                    and request.stream_options.include_usage):
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=previous_num_tokens[i],
                    total_tokens=prompt_tokens + previous_num_tokens[i],
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request],
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: List[ConversationMessage],
        tokenizer: PreTrainedTokenizer,
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = self.served_model_names[0]
        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices: List[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                )
            else:
                logprobs = None

            if request.tool_choice and type(
                    request.tool_choice) is ChatCompletionNamedToolChoiceParam:
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        ToolCall(function=FunctionCall(
                            name=request.tool_choice.function.name,
                            arguments=output.text))
                    ])
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, content=output.text)

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason)
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == role:
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

    def _get_top_logprobs(
            self, logprobs: Dict[int, Logprob], top_logprobs: Optional[int],
            tokenizer: PreTrainedTokenizer) -> List[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(token := self._get_decoded_token(p[1], p[0],
                                                        tokenizer)),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")))
            for i, p in enumerate(logprobs.items())
            if top_logprobs and i < top_logprobs
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[Dict[int, Logprob]]],
        tokenizer: PreTrainedTokenizer,
        num_output_top_logprobs: Optional[int] = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""

        logprobs_content = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace"))))
            else:
                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=step_top_logprobs[token_id].decoded_token,
                        logprob=max(step_top_logprobs[token_id].logprob,
                                    -9999.0),
                        bytes=list(
                            step_top_logprobs[token_id].decoded_token.encode(
                                "utf-8", errors="replace")),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs, num_output_top_logprobs,
                            tokenizer)))

        return ChatCompletionLogProbs(content=logprobs_content)
