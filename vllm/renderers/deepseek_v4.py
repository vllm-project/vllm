# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

from vllm.config import VllmConfig
from vllm.entrypoints.chat_utils import (
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

# The DeepSeek-V4 encoder joins the text parts of a tool result with a blank
# line (``"\n\n".join(text_parts)`` in ``deepseek_v4_encoding``); chat_utils
# would otherwise flatten them with a single ``"\n"`` before the encoder sees
# them, because most templates only accept string content for tool messages.
_TOOL_TEXT_SEPARATOR = "\n\n"


def _join_tool_text_parts(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Pre-join list-of-text ``tool`` results the way the encoder would."""
    out: list[ChatCompletionMessageParam] = []
    for message in messages:
        content = message.get("content")
        if (
            message.get("role") == "tool"
            and isinstance(content, list)
            and content
            and all(
                isinstance(part, dict) and part.get("type") == "text"
                for part in content
            )
        ):
            joined = _TOOL_TEXT_SEPARATOR.join(
                cast(dict, part).get("text", "") for part in content
            )
            message = cast(ChatCompletionMessageParam, {**message, "content": joined})
        out.append(message)
    return out


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
        return self.get_tokenizer().apply_chat_template(*args, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        messages = _join_tool_text_parts(messages)
        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
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
        messages = _join_tool_text_parts(messages)
        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
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
