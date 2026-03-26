# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeChatRequest,
    TokenizeRequest,
    TokenizeResponse,
    TokenizerInfoResponse,
)
from vllm.inputs import TokensPrompt, tokens_input
from vllm.inputs.engine import EngineInput
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


class OpenAIServingTokenization(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        openai_serving_render: OpenAIServingRender,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        trust_request_chat_template: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
        )

        self.openai_serving_render = openai_serving_render
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
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

        lora_request = self._maybe_get_adapters(request)

        if isinstance(request, TokenizeChatRequest):
            tool_dicts = (
                None
                if request.tools is None
                else [tool.model_dump() for tool in request.tools]
            )
            error_check_ret = self.openai_serving_render.validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )
            if error_check_ret is not None:
                return error_check_ret

            _, engine_inputs = await self.openai_serving_render.preprocess_chat(
                request,
                request.messages,
                default_template=self.chat_template,
                default_template_content_format=self.chat_template_content_format,
                default_template_kwargs=self.default_chat_template_kwargs,
                tool_dicts=tool_dicts,
            )
        else:
            engine_inputs = await self.openai_serving_render.preprocess_completion(
                request,
                prompt_input=request.prompt,
                prompt_embeds=None,
            )

        input_ids: list[int] = []
        for engine_input in engine_inputs:
            self._log_inputs(
                request_id,
                engine_input,
                params=None,
                lora_request=lora_request,
            )

            prompt_components = self._extract_prompt_components(engine_input)
            if prompt_components.token_ids is not None:
                input_ids.extend(prompt_components.token_ids)

        # The tokenize endpoint never sends multimodal data to the engine
        # via IPC, but the SenderCache already recorded these items as
        # "transmitted".  Evict newly-added entries so that a subsequent
        # generate request will re-process and actually transmit them.
        # Skip items where mm_kwargs is None — those were already sent
        # by a prior generate call and are valid in the receiver cache.
        self._evict_unsent_mm_cache_entries(engine_inputs)

        token_strs = None
        if request.return_token_strs:
            tokenizer = self.renderer.get_tokenizer()
            token_strs = tokenizer.convert_ids_to_tokens(input_ids)

        return TokenizeResponse(
            tokens=input_ids,
            token_strs=token_strs,
            count=len(input_ids),
            max_model_len=self.model_config.max_model_len,
        )

    def _evict_unsent_mm_cache_entries(
        self, engine_inputs: Sequence[EngineInput]
    ) -> None:
        """Remove sender-cache entries that were added but never sent via IPC.

        The tokenize endpoint fully processes multimodal inputs through the
        renderer, which populates the SenderCache.  Because the results are
        never forwarded to the engine, those entries would cause the next
        real generate call to receive ``None`` instead of tensor data.

        Only *newly added* items (``mm_kwargs[modality][idx] is not None``)
        are evicted; items that were already cached before this request
        (``None``) have been transmitted by a prior generate call and are
        still valid in the receiver cache.
        """
        mm_cache = self.openai_serving_render.renderer.mm_processor_cache
        if mm_cache is None:
            return

        for engine_input in engine_inputs:
            if engine_input.get("type") != "multimodal":
                continue

            mm_kwargs = engine_input.get("mm_kwargs", {})
            mm_hashes = engine_input.get("mm_hashes", {})

            for modality, hashes in mm_hashes.items():
                items = mm_kwargs.get(modality, ())
                for idx, h in enumerate(hashes):
                    if idx < len(items) and items[idx] is not None:
                        mm_cache.evict_item(h)

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

        self._log_inputs(
            request_id,
            tokens_input(request.tokens),
            params=None,
            lora_request=lora_request,
        )

        tok_prompt = await self.renderer.tokenize_prompt_async(
            TokensPrompt(prompt_token_ids=request.tokens),
            request.build_tok_params(self.model_config),
        )
        prompt_text = tok_prompt["prompt"]  # type: ignore[typeddict-item]

        return DetokenizeResponse(prompt=prompt_text)

    async def get_tokenizer_info(
        self,
    ) -> TokenizerInfoResponse | ErrorResponse:
        """Get comprehensive tokenizer information."""
        tokenizer = self.renderer.get_tokenizer()
        info = TokenizerInfo(tokenizer, self.chat_template).to_dict()
        return TokenizerInfoResponse(**info)


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
