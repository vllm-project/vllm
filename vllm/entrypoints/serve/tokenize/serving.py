# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any, Final

import jinja2
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeChatRequest,
    TokenizeRequest,
    TokenizeResponse,
    TokenizerInfoResponse,
)
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class OpenAIServingTokenization(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template

    async def create_tokenize(
        self,
        request: TokenizeRequest,
        raw_request: Request,
    ) -> TokenizeResponse | ErrorResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"tokenize-{self._base_request_id(raw_request)}"

        try:
            lora_request = self._maybe_get_adapters(request)

            if isinstance(request, TokenizeChatRequest):
                tool_dicts = (
                    None
                    if request.tools is None
                    else [tool.model_dump() for tool in request.tools]
                )
                error_check_ret = self._validate_chat_template(
                    request_chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs,
                    trust_request_chat_template=self.trust_request_chat_template,
                )
                if error_check_ret is not None:
                    return error_check_ret

                _, engine_prompts = await self._preprocess_chat(
                    request,
                    self.renderer,
                    request.messages,
                    default_template=self.chat_template,
                    default_template_content_format=self.chat_template_content_format,
                    default_template_kwargs=None,
                    tool_dicts=tool_dicts,
                )
            else:
                renderer = self._get_completion_renderer()
                engine_prompts = await renderer.render_prompt(
                    prompt_or_prompts=request.prompt,
                    config=self._build_render_config(request),
                )
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        input_ids: list[int] = []
        for engine_prompt in engine_prompts:
            self._log_inputs(
                request_id,
                engine_prompt,
                params=None,
                lora_request=lora_request,
            )

            if "prompt_token_ids" in engine_prompt:
                input_ids.extend(engine_prompt["prompt_token_ids"])  # type: ignore[typeddict-item]

        token_strs = None
        if request.return_token_strs:
            tokenizer = self.renderer.get_tokenizer()
            token_strs = tokenizer.convert_ids_to_tokens(input_ids)

        return TokenizeResponse(
            tokens=input_ids,
            token_strs=token_strs,
            count=len(input_ids),
            max_model_len=self.max_model_len,
        )

    async def create_detokenize(
        self,
        request: DetokenizeRequest,
        raw_request: Request,
    ) -> DetokenizeResponse | ErrorResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"tokenize-{self._base_request_id(raw_request)}"

        lora_request = self._maybe_get_adapters(request)
        tokenizer = self.renderer.get_tokenizer()

        self._log_inputs(
            request_id,
            TokensPrompt(prompt_token_ids=request.tokens),
            params=None,
            lora_request=lora_request,
        )

        engine_prompt = await self.renderer.tokenize_prompt_async(
            TokensPrompt(prompt_token_ids=request.tokens),
            request.build_tok_params(self.model_config),
        )
        prompt_text = engine_prompt["prompt"]  # type: ignore[typeddict-item]

        return DetokenizeResponse(prompt=prompt_text)

    async def get_tokenizer_info(
        self,
    ) -> TokenizerInfoResponse | ErrorResponse:
        """Get comprehensive tokenizer information."""
        try:
            tokenizer = self.renderer.get_tokenizer()
            info = TokenizerInfo(tokenizer, self.chat_template).to_dict()
            return TokenizerInfoResponse(**info)
        except Exception as e:
            return self.create_error_response(f"Failed to get tokenizer info: {str(e)}")


@dataclass
class TokenizerInfo:
    tokenizer: TokenizerLike
    chat_template: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the tokenizer configuration."""
        return self._get_tokenizer_config()

    def _get_tokenizer_config(self) -> dict[str, Any]:
        """Get tokenizer configuration directly from the tokenizer object."""
        config = dict(getattr(self.tokenizer, "init_kwargs", None) or {})

        # Remove file path fields
        config.pop("vocab_file", None)
        config.pop("merges_file", None)

        config = self._make_json_serializable(config)
        config["tokenizer_class"] = type(self.tokenizer).__name__
        if self.chat_template:
            config["chat_template"] = self.chat_template
        return config

    def _make_json_serializable(self, obj):
        """Convert any non-JSON-serializable objects to serializable format."""
        if hasattr(obj, "content"):
            return obj.content
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
