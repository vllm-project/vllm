# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
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
from vllm.inputs import ProcessorInputs, SingletonPrompt
from vllm.renderers import BaseRenderer, merge_kwargs
from vllm.renderers.inputs import TokPrompt
from vllm.renderers.inputs.preprocess import parse_model_prompt, prompt_to_seq
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils.mistral import is_mistral_tokenizer


class PoolingIOProcessor:
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        chat_template_config: ChatTemplateConfig,
    ):
        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)

        self.model_config = model_config
        self.renderer = renderer

        self.chat_template = chat_template_config.chat_template
        self.chat_template_content_format: Final = (
            chat_template_config.chat_template_content_format
        )
        self.trust_request_chat_template = (
            chat_template_config.trust_request_chat_template
        )

    def pre_process_online(self, *args, **kwargs):
        raise NotImplementedError

    async def pre_process_online_async(self, *args, **kwargs):
        return self.pre_process_online(*args, **kwargs)

    def pre_process_offline(self, *args, **kwargs):
        raise NotImplementedError

    async def pre_process_offline_async(self, *args, **kwargs):
        return self.pre_process_offline(*args, **kwargs)

    def post_process(
        self, outputs: list[PoolingRequestOutput]
    ) -> list[PoolingRequestOutput]:
        return outputs

    async def post_process_async(
        self, outputs: list[PoolingRequestOutput]
    ) -> list[PoolingRequestOutput]:
        return self.post_process(outputs)

    def create_pooling_params(self, request):
        return request.to_pooling_params()

    def _preprocess_completion_online(
        self,
        request: RendererRequest,
        prompt_input: str | list[str] | list[int] | list[list[int]] | None,
        prompt_embeds: bytes | list[bytes] | None,
    ) -> list[TokPrompt]:
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
        tool_parser: Callable[[TokenizerLike], ToolParser] | None = None,
    ) -> tuple[list[ConversationMessage], list[TokPrompt]]:
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

        (conversation,), (engine_prompt,) = renderer.render_chat(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
        )

        return conversation, [engine_prompt]

    def _preprocess_completion_offline(
        self,
        prompts: PromptType | Sequence[PromptType],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[ProcessorInputs]:
        renderer = self.renderer
        model_config = self.model_config

        prompts = prompt_to_seq(prompts)

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(model_config, prompt)
            )
            for prompt in prompts
        ]
        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        return renderer.render_cmpl(
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
