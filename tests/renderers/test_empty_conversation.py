# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An empty conversation must be rejected whatever the renderer.

The HF tokenizer refuses one inside ``apply_chat_template``, but renderers with
their own chat encoding (e.g. DeepSeek V3.2/V4/V4.1) render it to a bare BOS
prompt, from which the model generates unrelated text.
"""

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.renderers.base import BaseRenderer
from vllm.renderers.params import ChatParams, TokenizeParams


class _Rendered(Exception):
    """Raised by the stub once a conversation reaches ``render_messages``."""


def _renderer() -> BaseRenderer:
    class _StubRenderer(BaseRenderer):
        def __init__(self):
            self.tokenizer = None

        def render_messages(self, messages, params):
            raise _Rendered

    return _StubRenderer()


def _render(renderer, conversations):
    return renderer.render_chat(
        conversations, ChatParams(), TokenizeParams(max_total_tokens=4096)
    )


@pytest.mark.parametrize(
    "conversations",
    [[[]], [[{"role": "user", "content": "hi"}], []]],
    ids=["only", "one_of_two"],
)
def test_render_chat_rejects_an_empty_conversation(conversations):
    with pytest.raises(VLLMValidationError, match="empty conversation"):
        _render(_renderer(), conversations)


@pytest.mark.asyncio
async def test_render_chat_async_rejects_an_empty_conversation():
    with pytest.raises(VLLMValidationError, match="empty conversation"):
        await _renderer().render_chat_async(
            [[]], ChatParams(), TokenizeParams(max_total_tokens=4096)
        )


@pytest.mark.parametrize(
    "conversation",
    [[{"role": "user", "content": "hi"}], [{"role": "system", "content": "Be brief."}]],
    ids=["user", "system_only"],
)
def test_render_chat_renders_a_non_empty_conversation(conversation):
    with pytest.raises(_Rendered):
        _render(_renderer(), [conversation])
