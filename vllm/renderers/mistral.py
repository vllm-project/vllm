# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from typing_extensions import override

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


def _get_chat_template_kwargs(params: ChatParams) -> dict[str, Any]:
    kwargs = params.get_apply_chat_template_kwargs()
    if params.response_format is not None:
        rf = params.response_format.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        # mistral-common only accepts JSON objects and JSON schemas.
        if rf["type"] == "json_object" or "schema" in rf.get("json_schema", {}):
            kwargs["response_format"] = rf
    return kwargs


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
            tokenizer, messages, **_get_chat_template_kwargs(params)
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
            tokenizer, messages, **_get_chat_template_kwargs(params)
        )

        prompt = parse_dec_only_prompt(prompt_raw)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids

        return conversation, prompt

    @override
    def _tokenize_prompt(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokens_prompt = super()._tokenize_prompt(prompt, params)
        if not self._keeps_special_tokens(prompt, params):
            return tokens_prompt

        # With an explicit side, the tokenizer does not truncate the prompt (see
        # `TokenizeParams.get_encode_kwargs`), so truncate it here and keep BOS,
        # the only token that mistral-common adds, as its own truncation does.
        tokenizer = self.get_tokenizer()
        token_ids = tokens_prompt["prompt_token_ids"]
        truncation = params._truncation_slice(tokenizer, len(token_ids))
        if truncation is not None and token_ids[:1] == [tokenizer.bos_token_id]:
            special_tokens_mask = [1] + [0] * (len(token_ids) - 1)
            kept = params._truncation_indices(truncation, special_tokens_mask)
            if kept is not None:
                tokens_prompt["prompt_token_ids"] = [token_ids[i] for i in kept]

        return tokens_prompt
