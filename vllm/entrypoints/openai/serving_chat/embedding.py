import asyncio
import time
from typing import List, Optional, Union

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         load_chat_template,
                                         parse_chat_messages_futures)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ChatEmbeddingRequest,
                                              EmbeddingResponse, ErrorResponse)
from vllm.entrypoints.openai.serving_embedding import (
    check_embedding_mode, request_output_to_embedding_response)
from vllm.entrypoints.openai.serving_engine import (BaseModelPath, OpenAIServing,
                                                    TextTokensPrompt)
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import EmbeddingRequestOutput
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import iterate_with_cancellation, is_list_of, random_uuid

logger = init_logger(__name__)


class OpenAIServingChatEmbedding(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        base_model_paths: List[BaseModelPath],
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         base_model_paths=base_model_paths,
                         lora_modules=None,
                         prompt_adapters=None,
                         request_logger=request_logger)

        self.chat_template = load_chat_template(chat_template)

        self._enabled = check_embedding_mode(model_config)

    async def create_embedding(
        self,
        request: ChatEmbeddingRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[EmbeddingResponse, ErrorResponse]:
        """
        Chat Embedding API, a variant of Embedding API that accepts chat conversations
        which can include multi-modal data. 
        """
        if not self._enabled:
            return self.create_error_response("Embedding API disabled")
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        encoding_format = request.encoding_format
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = request.model
        request_id = f"chatembd-{random_uuid()}"
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

            model_config = self.model_config
            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            conversation, mm_data_future = parse_chat_messages_futures(
                request.messages, model_config, tokenizer)

            prompt: Union[str, List[int]]
            is_mistral_tokenizer = isinstance(tokenizer, MistralTokenizer)
            if is_mistral_tokenizer:
                prompt = apply_mistral_chat_template(
                    tokenizer,
                    messages=request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    add_generation_prompt=request.add_generation_prompt,
                    **(request.chat_template_kwargs or {}),
                )
            else:
                prompt = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=request.chat_template or self.chat_template,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    **(request.chat_template_kwargs or {}),
                )
        except Exception as e:
            logger.exception("Error in applying chat template from request")
            return self.create_error_response(str(e))

        try:
            mm_data = await mm_data_future
        except Exception as e:
            logger.exception("Error in loading multi-modal data")
            return self.create_error_response(str(e))

        try:
            pooling_params = request.to_pooling_params()

            if isinstance(prompt, str):
                prompt_inputs = self._tokenize_prompt_input(
                    request,
                    tokenizer,
                    prompt,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                # For MistralTokenizer
                assert is_list_of(prompt, int), (
                    "Prompt has to be either a string or a list of token ids")
                prompt_inputs = TextTokensPrompt(
                    prompt=tokenizer.decode(prompt), prompt_token_ids=prompt)

            assert prompt_inputs is not None

            self._log_inputs(request_id,
                             prompt_inputs,
                             params=pooling_params,
                             lora_request=lora_request,
                             prompt_adapter_request=prompt_adapter_request)

            engine_inputs = TokensPrompt(
                prompt_token_ids=prompt_inputs["prompt_token_ids"])
            if mm_data is not None:
                engine_inputs["multi_modal_data"] = mm_data

            result_generator = self.engine_client.encode(
                engine_inputs,
                pooling_params,
                request_id,
                lora_request=lora_request,
                priority=request.priority,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        if raw_request:
            result_generator = iterate_with_cancellation(
                result_generator, raw_request.is_disconnected)

        # Non-streaming response
        final_res: Optional[EmbeddingRequestOutput] = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        assert final_res is not None

        try:
            response = request_output_to_embedding_response(
                [final_res], request_id, created_time, model_name,
                encoding_format)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response
