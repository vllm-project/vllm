# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormat,
    ConversationMessage,
    apply_step_audio_2_chat_template,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers import cached_get_tokenizer
from vllm.tokenizers.step_audio_2 import StepAudio2Tokenizer

from .base import BaseRenderer
from .hf import resolve_chat_template_content_format
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams


class StepAudio2Renderer(BaseRenderer[StepAudio2Tokenizer]):
    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: VllmConfig,
        tokenizer_kwargs: dict[str, Any],
    ) -> "StepAudio2Renderer":
        model_config = config.model_config
        if model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_get_tokenizer(
                tokenizer_cls=StepAudio2Tokenizer,
                **tokenizer_kwargs,
            )

        return cls(config, tokenizer)

    def _resolve_content_format(
        self, tokenizer: StepAudio2Tokenizer, params: ChatParams
    ) -> ChatTemplateContentFormat:
        return resolve_chat_template_content_format(
            chat_template=params.chat_template,
            tools=params.chat_template_kwargs.get("tools"),
            given_format=params.chat_template_content_format,
            tokenizer=tokenizer,
            model_config=self.model_config,
        )

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format=self._resolve_content_format(tokenizer, params),
        )

        prompt_raw = apply_step_audio_2_chat_template(
            tokenizer,
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format=self._resolve_content_format(tokenizer, params),
        )

        prompt_raw = apply_step_audio_2_chat_template(
            tokenizer,
            conversation,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
