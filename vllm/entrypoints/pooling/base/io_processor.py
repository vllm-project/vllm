# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any, Final

from vllm import PoolingRequestOutput, PromptType
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateConfig,
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.openai.engine.serving import RendererChatRequest, RendererRequest
from vllm.entrypoints.pooling.typing import (
    OfflineInputsContext,
    OfflineOutputsContext,
    PoolingChatLikeRequest,
    PoolingCompletionLikeRequest,
    PoolingServeContext,
)
from vllm.inputs import EngineInput, SingletonPrompt
from vllm.renderers import BaseRenderer, TokenizeParams, merge_kwargs
from vllm.renderers.inputs.preprocess import parse_model_prompt, prompt_to_seq
from vllm.tool_parsers import ToolParser
from vllm.utils.mistral import is_mistral_tokenizer


class PoolingIOProcessor:
    name: str

    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ):
        self.model_config = model_config
        self.renderer = renderer

        self.chat_template = chat_template_config.chat_template
        self.chat_template_content_format: Final = (
            chat_template_config.chat_template_content_format
        )
        self.trust_request_chat_template = (
            chat_template_config.trust_request_chat_template
        )

    def create_pooling_params(self, request):
        return request.to_pooling_params()

    #######################################
    # online APIs

    def pre_process_online(self, ctx: PoolingServeContext):
        request = ctx.request

        if isinstance(ctx.request, PoolingChatLikeRequest):
            self._validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )
            _, engine_inputs = self._preprocess_chat_online(
                request,
                request.messages,
                default_template=self.chat_template,
                default_template_content_format=self.chat_template_content_format,
                default_template_kwargs=None,
            )
        elif isinstance(request, PoolingCompletionLikeRequest):
            engine_inputs = self._preprocess_completion_online(
                request,
                prompt_input=request.input,
                prompt_embeds=None,
            )
        else:
            raise ValueError(f"Invalid {self.name} request type")

        ctx.engine_inputs = engine_inputs

    async def pre_process_online_async(self, ctx: PoolingServeContext):
        self.pre_process_online(ctx)

    def post_process_online(
        self,
        ctx: PoolingServeContext,
    ):
        pass

    async def post_process_online_async(
        self,
        ctx: PoolingServeContext,
    ):
        self.post_process_online(ctx)

    #######################################
    # offline APIs

    def pre_process_offline(self, ctx: OfflineInputsContext) -> Sequence[EngineInput]:
        tok_params = self.renderer.default_cmpl_tok_params.with_kwargs(
            **(ctx.tokenization_kwargs or {})
        )
        return self._preprocess_completion_offline(
            prompts=ctx.prompts, tok_params=tok_params
        )

    async def pre_process_offline_async(self, ctx: OfflineInputsContext):
        return self.pre_process_offline(ctx)

    def post_process_offline(
        self,
        ctx: OfflineOutputsContext,
    ) -> list[PoolingRequestOutput]:
        return ctx.outputs

    async def post_process_offline_async(
        self,
        ctx: OfflineOutputsContext,
    ) -> list[PoolingRequestOutput]:
        return self.post_process_offline(ctx)

    #######################################
    # helpers

    def _preprocess_completion_online(
        self,
        request: RendererRequest,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
    ) -> list[EngineInput]:
        renderer = self.renderer
        model_config = self.model_config

        prompts = list[SingletonPrompt | bytes]()
        if prompt_embeds is not None:  # embeds take higher priority
            prompts.extend(prompt_to_seq(prompt_embeds))
        if prompt_input is not None:
            prompts.extend(prompt_to_seq(prompt_input))

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(model_config, prompt)
            )
            for prompt in prompts
        ]
        tok_params = request.build_tok_params(model_config)

        return renderer.render_cmpl(
            parsed_prompts,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

    def _preprocess_chat_online(
        self,
        request: RendererChatRequest,
        messages: list[ChatCompletionMessageParam],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        tool_parser: type[ToolParser] | None = None,
    ) -> tuple[list[ConversationMessage], list[EngineInput]]:
        renderer = self.renderer

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=is_mistral_tokenizer(renderer.tokenizer),
            ),
        )

        mm_config = self.model_config.multimodal_config

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
        )

        (conversation,), (engine_input,) = renderer.render_chat(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

        return conversation, [engine_input]

    def _preprocess_completion_offline(
        self,
        prompts: PromptType | Sequence[PromptType],
        tok_params: TokenizeParams,
    ) -> Sequence[EngineInput]:
        prompts = prompt_to_seq(prompts)
        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(self.model_config, prompt)
            )
            for prompt in prompts
        ]

        return self.renderer.render_cmpl(
            parsed_prompts,
            tok_params,
        )

    def _validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ):
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            raise ValueError(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None
