# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams

_FIM_BEGIN = "<｜fim▁begin｜>"
_FIM_HOLE = "<｜fim▁hole｜>"
_FIM_END = "<｜fim▁end｜>"


class DeepseekV4Renderer(BaseRenderer[DeepseekV4Tokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: DeepseekV4Tokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            self._apply_chat_template, executor=self._executor
        )

    def _apply_chat_template(self, *args, **kwargs):
        return self.get_tokenizer().apply_chat_template(*args, **kwargs), None

    def _build_prompt(self, rendered, mm_data, mm_uuids) -> DictPrompt:
        prompt_raw, image_order = rendered
        prompt = parse_dec_only_prompt(prompt_raw)
        if image_order is not None:
            for items in (mm_data, mm_uuids):
                if items is not None and "image" in items:
                    if image_order:
                        items["image"] = [items["image"][i] for i in image_order]
                    else:
                        del items["image"]
        if mm_data:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids:
            prompt["multi_modal_uuids"] = mm_uuids
        return prompt

    def render_completion_suffix(self, prompt: str, suffix: str) -> str | None:
        return f"{_FIM_BEGIN}{prompt}{_FIM_HOLE}{suffix}{_FIM_END}"

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered = self._apply_chat_template(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        return conversation, self._build_prompt(rendered, mm_data, mm_uuids)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        rendered = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        return conversation, self._build_prompt(rendered, mm_data, mm_uuids)


class DeepseekV41Renderer(DeepseekV4Renderer):
    def _apply_chat_template(self, *args, **kwargs):
        image_order: list[int] = []
        kwargs["image_order"] = image_order
        prompt_raw = self.get_tokenizer().apply_chat_template(*args, **kwargs)
        return prompt_raw, image_order
