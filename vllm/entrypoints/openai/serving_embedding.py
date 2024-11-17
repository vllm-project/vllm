import asyncio
import base64
import time
from typing import (Annotated, Any, AsyncGenerator, Dict, Final, List, Literal,
                    Optional, Sequence, Tuple, Union, cast)

import numpy as np
from fastapi import Request
from pydantic import Field
from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         ConversationMessage,
                                         parse_chat_messages_futures)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (EmbeddingChatRequest,
                                              EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData,
                                              ErrorResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (BaseModelPath,
                                                    OpenAIServing,
                                                    RequestPrompt)
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import EmbeddingOutput, EmbeddingRequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from vllm.utils import merge_async_iterators, random_uuid

logger = init_logger(__name__)


def _get_embedding(
    output: EmbeddingOutput,
    encoding_format: Literal["float", "base64"],
) -> Union[List[float], str]:
    if encoding_format == "float":
        return output.embedding
    elif encoding_format == "base64":
        # Force to use float32 for base64 encoding
        # to match the OpenAI python client behavior
        embedding_bytes = np.array(output.embedding, dtype="float32").tobytes()
        return base64.b64encode(embedding_bytes).decode("utf-8")

    assert_never(encoding_format)


def request_output_to_embedding_response(
        final_res_batch: List[EmbeddingRequestOutput], request_id: str,
        created_time: int, model_name: str,
        encoding_format: Literal["float", "base64"]) -> EmbeddingResponse:
    data: List[EmbeddingResponseData] = []
    num_prompt_tokens = 0
    for idx, final_res in enumerate(final_res_batch):
        prompt_token_ids = final_res.prompt_token_ids
        embedding = _get_embedding(final_res.outputs, encoding_format)
        embedding_data = EmbeddingResponseData(index=idx, embedding=embedding)
        data.append(embedding_data)

        num_prompt_tokens += len(prompt_token_ids)

    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        total_tokens=num_prompt_tokens,
    )

    return EmbeddingResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        data=data,
        usage=usage,
    )


class OpenAIServingEmbedding(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        base_model_paths: List[BaseModelPath],
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         base_model_paths=base_model_paths,
                         lora_modules=None,
                         prompt_adapters=None,
                         request_logger=request_logger)

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

    async def create_embedding(
        self,
        request: EmbeddingRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[EmbeddingResponse, ErrorResponse]:
        """
        Embedding API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        encoding_format = request.encoding_format
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = request.model
        request_id = f"embd-{random_uuid()}"
        created_time = int(time.monotonic())

        truncate_prompt_tokens = None

        if request.truncate_prompt_tokens is not None:
            if request.truncate_prompt_tokens <= self.max_model_len:
                truncate_prompt_tokens = request.truncate_prompt_tokens
            else:
                return self.create_error_response(
                    "truncate_prompt_tokens value is "
                    "greater than max_model_len."
                    " Please, select a smaller truncation size.")

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for embedding models")

            if self.model_config.task == "cross_encoding":
                if isinstance(request, EmbeddingChatRequest):
                    (
                        _,
                        request_prompts,
                        engine_prompts,
                    ) = await self._preprocess_cross_encoding(
                        tokenizer,
                        request.messages,
                        truncate_prompt_tokens=truncate_prompt_tokens,
                    )
                else:
                    return self.create_error_response(
                        "Cross encoding requests must "
                        "use the chat embedding API")
            else:
                if isinstance(request, EmbeddingChatRequest):
                    (
                        _,
                        request_prompts,
                        engine_prompts,
                    ) = await self._preprocess_chat(
                        request,
                        tokenizer,
                        request.messages,
                        chat_template=request.chat_template
                        or self.chat_template,
                        chat_template_content_format=self.
                        chat_template_content_format,
                        add_generation_prompt=request.add_generation_prompt,
                        continue_final_message=request.continue_final_message,
                        truncate_prompt_tokens=truncate_prompt_tokens,
                        add_special_tokens=request.add_special_tokens,
                    )
                else:
                    request_prompts, engine_prompts = self\
                        ._preprocess_completion(
                        request,
                        tokenizer,
                        request.input,
                        truncate_prompt_tokens=truncate_prompt_tokens,
                        add_special_tokens=request.add_special_tokens,
                    )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[EmbeddingRequestOutput, None]] = []
        try:
            pooling_params = request.to_pooling_params()

            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(
            *generators,
            is_cancelled=raw_request.is_disconnected if raw_request else None,
        )

        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: List[Optional[EmbeddingRequestOutput]]
        final_res_batch = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(List[EmbeddingRequestOutput],
                                           final_res_batch)

            response = request_output_to_embedding_response(
                final_res_batch_checked, request_id, created_time, model_name,
                encoding_format)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response

    async def _preprocess_cross_encoding(
        self,
        tokenizer: AnyTokenizer,
        messages: List[ChatCompletionMessageParam],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=3)]] = None,
    ) -> Tuple[List[ConversationMessage], Sequence[RequestPrompt],
               List[TokensPrompt]]:

        conversation, mm_data_future = parse_chat_messages_futures(
            messages, self.model_config, tokenizer, "string")
        await mm_data_future

        if len(conversation) != 2:
            raise ValueError("For cross encoding two inputs must be provided")
        prompts = []
        for msg in conversation:
            content = msg["content"]
            assert type(msg["content"]) is str
            prompts.append(content)

        if isinstance(tokenizer, MistralTokenizer):
            raise ValueError(
                "MistralTokenizer not supported for cross-encoding")

        request_prompt = f"{prompts[0]}{tokenizer.sep_token}{prompts[1]}"

        tokenization_kwargs: Dict[str, Any] = {}
        if truncate_prompt_tokens is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = truncate_prompt_tokens

        prompt_inputs = tokenizer(prompts[0],
                                  text_pair=prompts[1],
                                  **tokenization_kwargs)

        engine_prompt = TokensPrompt(
            prompt_token_ids=prompt_inputs["input_ids"],
            token_type_ids=prompt_inputs["token_type_ids"])

        return conversation, [request_prompt], [engine_prompt]
