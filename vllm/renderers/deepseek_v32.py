# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers.deepseek_v32 import DeepseekV32Tokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams, TokenizeParams


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

    def _tokenize_prompt(
        self, prompt: TextPrompt, params: TokenizeParams
    ) -> TokensPrompt:
        if not (
            params.needs_detokenization
            and params.add_special_tokens
            and params.truncate_prompt_tokens is not None
            and params.truncation_side is not None
        ):
            return super()._tokenize_prompt(prompt, params)

        tokenizer = self.get_tokenizer()
        want_offsets = self._wants_offsets(prompt, params)
        kwargs = params.get_encode_kwargs()
        kwargs["return_special_tokens_mask"] = True
        if want_offsets:
            kwargs["return_offsets_mapping"] = True
        try:
            encoding = tokenizer(prompt["prompt"], **kwargs)
        except (TypeError, NotImplementedError):
            return super()._tokenize_prompt(prompt, params)

        tokenized = self._build_tokens_prompt(
            encoding["input_ids"],
            prompt,
            offset_mapping=encoding["offset_mapping"] if want_offsets else None,
        )
        token_ids = tokenized["prompt_token_ids"]
        if (
            (mask := encoding.get("special_tokens_mask")) is not None
            and len(mask) == len(token_ids)
            and params._truncation_slice(tokenizer, len(token_ids)) is not None
        ):
            tokenized["prompt_special_tokens_mask"] = mask
        return tokenized

    def _apply_chat_template(self, *args, **kwargs):
        return self.get_tokenizer().apply_chat_template(*args, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = self._apply_chat_template(
            conversation=conversation,
            messages=messages,
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
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = await self._apply_chat_template_async(
            conversation=conversation,
            messages=messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
