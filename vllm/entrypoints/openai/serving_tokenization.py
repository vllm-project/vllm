from typing import List, Optional, Union

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         load_chat_template,
                                         parse_chat_messages_futures)
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (DetokenizeRequest,
                                              DetokenizeResponse,
                                              ErrorResponse,
                                              TokenizeChatRequest,
                                              TokenizeRequest,
                                              TokenizeResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_engine import (BaseModelPath,
                                                    LoRAModulePath,
                                                    OpenAIServing)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


class OpenAIServingTokenization(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        base_model_paths: List[BaseModelPath],
        *,
        lora_modules: Optional[List[LoRAModulePath]],
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         base_model_paths=base_model_paths,
                         lora_modules=lora_modules,
                         prompt_adapters=None,
                         request_logger=request_logger)

        # If this is None we use the tokenizer's default chat template
        # the list of commonly-used chat template names for HF named templates
        hf_chat_templates: List[str] = ['default', 'tool_use']
        self.chat_template = chat_template \
            if chat_template in hf_chat_templates \
            else load_chat_template(chat_template)

    async def create_tokenize(
        self,
        request: TokenizeRequest,
    ) -> Union[TokenizeResponse, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"tokn-{random_uuid()}"

        (
            lora_request,
            prompt_adapter_request,
        ) = self._maybe_get_adapters(request)

        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        prompt: Union[str, List[int]]
        if isinstance(request, TokenizeChatRequest):
            model_config = self.model_config

            conversation, mm_data_future = parse_chat_messages_futures(
                request.messages, model_config, tokenizer)

            mm_data = await mm_data_future
            if mm_data:
                logger.warning(
                    "Multi-modal inputs are ignored during tokenization")

            if isinstance(tokenizer, MistralTokenizer):
                prompt = apply_mistral_chat_template(
                    tokenizer,
                    messages=request.messages,
                    chat_template=self.chat_template,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                )
            else:
                prompt = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=self.chat_template,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                )
        else:
            prompt = request.prompt

        self._log_inputs(request_id,
                         prompt,
                         params=None,
                         lora_request=lora_request,
                         prompt_adapter_request=prompt_adapter_request)

        # Silently ignore prompt adapter since it does not affect tokenization

        prompt_input = self._tokenize_prompt_input(
            request,
            tokenizer,
            prompt,
            add_special_tokens=request.add_special_tokens,
        )
        input_ids = prompt_input["prompt_token_ids"]

        return TokenizeResponse(tokens=input_ids,
                                count=len(input_ids),
                                max_model_len=self.max_model_len)

    async def create_detokenize(
        self,
        request: DetokenizeRequest,
    ) -> Union[DetokenizeResponse, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"tokn-{random_uuid()}"

        (
            lora_request,
            prompt_adapter_request,
        ) = self._maybe_get_adapters(request)

        tokenizer = await self.engine_client.get_tokenizer(lora_request)

        self._log_inputs(request_id,
                         request.tokens,
                         params=None,
                         lora_request=lora_request,
                         prompt_adapter_request=prompt_adapter_request)

        if prompt_adapter_request is not None:
            raise NotImplementedError("Prompt adapter is not supported "
                                      "for tokenization")

        prompt_input = self._tokenize_prompt_input(
            request,
            tokenizer,
            request.tokens,
        )
        input_text = prompt_input["prompt"]

        return DetokenizeResponse(prompt=input_text)
