from typing import List, Optional

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.chat_utils import (ConversationMessage,
                                         load_chat_template,
                                         parse_chat_message_content)
from vllm.entrypoints.openai.protocol import (DetokenizeRequest,
                                              DetokenizeResponse,
                                              TokenizeRequest,
                                              TokenizeResponse)
from vllm.entrypoints.openai.serving_engine import (LoRAModulePath,
                                                    OpenAIServing)


class OpenAIServingTokenization(OpenAIServing):

    def __init__(self,
                 engine: AsyncLLMEngine,
                 model_config: ModelConfig,
                 served_model_names: List[str],
                 lora_modules: Optional[List[LoRAModulePath]] = None,
                 chat_template: Optional[str] = None):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=lora_modules)

        # If this is None we use the tokenizer's default chat template
        self.chat_template = load_chat_template(chat_template)

    async def create_tokenize(self,
                              request: TokenizeRequest) -> TokenizeResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if not (request.prompt or request.messages):
            return self.create_error_response(
                "Either `prompt` or `messages` should be provided.")

        if (request.prompt and request.messages):
            return self.create_error_response(
                "Only one of `prompt` or `messages` should be provided.")

        _, lora_request = self._maybe_get_adapter(request)
        tokenizer = await self.engine.get_tokenizer(lora_request)
        if request.messages:
            conversation: List[ConversationMessage] = []

            for message in request.messages:
                result = parse_chat_message_content(message, self.model_config,
                                                    tokenizer)
                conversation.extend(result.messages)

            request.prompt = tokenizer.apply_chat_template(
                add_generation_prompt=request.add_generation_prompt,
                conversation=conversation,
                tokenize=False,
                chat_template=self.chat_template)

        (input_ids, input_text) = await self._validate_prompt_and_tokenize(
            request,
            tokenizer,
            prompt=request.prompt,
            add_special_tokens=request.add_special_tokens)

        return TokenizeResponse(tokens=input_ids,
                                count=len(input_ids),
                                max_model_len=self.max_model_len)

    async def create_detokenize(
            self, request: DetokenizeRequest) -> DetokenizeResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        _, lora_request = self._maybe_get_adapter(request)
        tokenizer = await self.engine.get_tokenizer(lora_request)
        (input_ids, input_text) = await self._validate_prompt_and_tokenize(
            request, tokenizer, prompt_ids=request.tokens)

        return DetokenizeResponse(prompt=input_text)
