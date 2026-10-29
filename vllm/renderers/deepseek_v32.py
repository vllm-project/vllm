# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, cast

from vllm.config import VllmConfig
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.hf import (
    resolve_chat_template_content_format,
    safe_apply_chat_template,
)
from vllm.tokenizers.deepseek_v32 import DeepseekV32Tokenizer
from vllm.tokenizers.hf import HfTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

if TYPE_CHECKING:
    from vllm.renderers.chat_utils import ChatTemplateContentFormat


class DeepseekV32Renderer(BaseRenderer[DeepseekV32Tokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: DeepseekV32Tokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )
        self._safe_apply_chat_template_async = make_async(
            safe_apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(self, *args, **kwargs):
        return self.get_tokenizer().apply_chat_template(*args, **kwargs)

    @staticmethod
    def _uses_custom_template(params: ChatParams) -> bool:
        """Whether the request overrides the built-in DeepSeek V3.2 encoding."""
        return params.chat_template is not None

    def _content_format(self, params: ChatParams) -> "ChatTemplateContentFormat":
        if not self._uses_custom_template(params):
            # The built-in DeepSeek V3.2 encoding consumes plain strings.
            return "string"
        return resolve_chat_template_content_format(
            chat_template=params.chat_template,
            tools=params.chat_template_kwargs.get("tools"),
            given_format=params.chat_template_content_format,
            tokenizer=cast(HfTokenizer, self.get_tokenizer()),
            model_config=self.model_config,
        )

    def _render_prompt(
        self,
        conversation: list[ConversationMessage],
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> str | list[int]:
        chat_template_kwargs = params.get_apply_chat_template_kwargs()
        if not self._uses_custom_template(params):
            return self._apply_chat_template(
                conversation=conversation,
                messages=messages,
                **chat_template_kwargs,
            )
        return safe_apply_chat_template(
            self.model_config,
            cast(HfTokenizer, self.get_tokenizer()),
            conversation,
            **chat_template_kwargs,
        )

    async def _render_prompt_async(
        self,
        conversation: list[ConversationMessage],
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> str | list[int]:
        chat_template_kwargs = params.get_apply_chat_template_kwargs()
        if not self._uses_custom_template(params):
            return await self._apply_chat_template_async(
                conversation=conversation,
                messages=messages,
                **chat_template_kwargs,
            )
        return await self._safe_apply_chat_template_async(
            self.model_config,
            cast(HfTokenizer, self.get_tokenizer()),
            conversation,
            **chat_template_kwargs,
        )

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format=self._content_format(params),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt = parse_dec_only_prompt(
            self._render_prompt(conversation, messages, params)
        )
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
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format=self._content_format(params),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt = parse_dec_only_prompt(
            await self._render_prompt_async(conversation, messages, params)
        )
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
