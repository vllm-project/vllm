# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from dataclasses import replace

from vllm.renderers.chat_utils import ChatCompletionMessageParam, ConversationMessage

from .hf import HfRenderer
from .inputs import DictPrompt
from .params import ChatParams


def _exclude_tools_for_none(params: ChatParams) -> ChatParams:
    if params.chat_template is not None or params.tool_choice != "none":
        return params
    return replace(
        params, chat_template_kwargs={**params.chat_template_kwargs, "tools": []}
    )


class GigaChat4Renderer(HfRenderer):
    """Render GigaChat4 messages with native tool-choice handling."""

    def render_messages(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        params = _exclude_tools_for_none(params)
        if params.chat_template is None:
            messages = deepcopy(messages)
        return super().render_messages(messages, params)

    async def render_messages_async(
        self, messages: list[ChatCompletionMessageParam], params: ChatParams
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        params = _exclude_tools_for_none(params)
        if params.chat_template is None:
            messages = deepcopy(messages)
        return await super().render_messages_async(messages, params)
