from typing import List, Optional

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import AsyncEngineClient
from vllm.entrypoints.chat_utils import load_chat_template, parse_chat_messages
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import CachingRequest, CachingResponse
# yapf: enable
from vllm.entrypoints.openai.serving_engine import (LoRAModulePath,
                                                    OpenAIServing,
                                                    PromptAdapterPath)
from vllm.inputs import PromptInputs
from vllm.logger import init_logger
from vllm.tracing import contains_trace_headers, log_tracing_disabled_warning
from vllm.utils import random_uuid

logger = init_logger(__name__)


class OpenAIServingCaching(OpenAIServing):

    def __init__(
        self,
        async_engine_client: AsyncEngineClient,
        model_config: ModelConfig,
        served_model_names: List[str],
        response_role: str,
        *,
        lora_modules: Optional[List[LoRAModulePath]],
        prompt_adapters: Optional[List[PromptAdapterPath]],
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(async_engine_client=async_engine_client,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=lora_modules,
                         prompt_adapters=prompt_adapters,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        self.response_role = response_role

        # If this is None we use the tokenizer's default chat template
        self.chat_template = load_chat_template(chat_template)

    async def create_caching(self, request: CachingRequest,
                             raw_request: Request) -> CachingResponse:
        """Context caching API similar to kimi's caching API.

        See https://platform.moonshot.cn/docs/api/caching 
        for the API specification. This API mimics the Kimi Caching API.

        TODO
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"cach-{random_uuid()}"

        try:
            model_config = self.model_config
            tokenizer = await self.async_engine_client.get_tokenizer(None)

            conversation, mm_futures = parse_chat_messages(
                request.messages, model_config, tokenizer)

            if mm_futures is not None and len(mm_futures) > 0:  # TODO
                raise NotImplementedError(
                    "Context caching with mm is not supported")

            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]

            if tool_dicts is not None:  # TODO
                raise NotImplementedError(
                    "Context caching with tools is not supported")

            prompt = tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                chat_template=self.chat_template)
            assert isinstance(prompt, str)
        except Exception as e:
            logger.error("Error in applying chat template for request")
            return self.create_error_response(str(e))

        # TODO mm_data

        request_id = f"cache-{random_uuid()}"
        try:
            prompt_inputs = self._tokenize_prompt_input(
                request,
                tokenizer,
                prompt,
            )

            engine_inputs: PromptInputs = {
                "prompt_token_ids": prompt_inputs["prompt_token_ids"],
            }

            is_tracing_enabled = (
                await self.async_engine_client.is_tracing_enabled())

            if (not is_tracing_enabled and raw_request
                    and contains_trace_headers(raw_request.headers)):
                log_tracing_disabled_warning()

            response = await self.async_engine_client.caching(
                inputs=engine_inputs,
                request_id=request_id,
                expired_at=request.expired_at,
                ttl=request.ttl)

            return CachingResponse(id=response.id,
                                   status=response.status,
                                   object=response.object,
                                   created_at=response.created_at,
                                   expired_at=response.expired_at,
                                   tokens=response.tokens,
                                   error=response.error)
            # Process response

        except ValueError as e:
            return self.create_error_response(str(e))
