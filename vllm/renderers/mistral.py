# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.logger import init_logger
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.utils.async_utils import make_async

from .base import BaseRenderer
from .inputs import DictPrompt
from .inputs.preprocess import parse_dec_only_prompt
from .params import ChatParams, TokenizeParams

logger = init_logger(__name__)


def safe_apply_chat_template(
    tokenizer: MistralTokenizer,
    messages: list[ChatCompletionMessageParam],
    **kwargs,
) -> str | list[int]:
    from mistral_common.exceptions import MistralCommonException

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    # mistral-common uses assert statements to stop processing of input
    # if input does not comply with the expected format.
    # We convert those assertion errors to ValueErrors so they can be
    # properly caught in the preprocessing_input step
    except (AssertionError, MistralCommonException) as e:
        raise ValueError(str(e)) from e

    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except Exception as e:
        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `mistral_common` while applying chat template"
        )
        raise ValueError(str(e)) from e


class MistralRenderer(BaseRenderer[MistralTokenizer]):
    def __init__(
        self,
        config: VllmConfig,
        tokenizer: MistralTokenizer | None,
    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
            safe_apply_chat_template, executor=self._executor
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

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        tokenizer = self.get_tokenizer()
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = safe_apply_chat_template(
            tokenizer,
            messages,
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
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )

        prompt_raw = await self._apply_chat_template_async(
            tokenizer,
            messages,
            **params.get_apply_chat_template_kwargs(),
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt
