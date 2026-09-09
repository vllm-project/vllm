# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.renderers.deepseek_v4 as deepseek_v4_renderer
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer


def test_deepseek_v4_renderer_keeps_structured_multimodal_content(monkeypatch):
    messages = [{"role": "user", "content": "hello"}]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "after"},
            ],
        }
    ]
    calls = {}

    def fake_parse_chat_messages(
        messages_arg,
        model_config,
        content_format,
        media_io_kwargs=None,
        mm_processor_kwargs=None,
    ):
        calls["messages"] = messages_arg
        calls["model_config"] = model_config
        calls["content_format"] = content_format
        calls["media_io_kwargs"] = media_io_kwargs
        calls["mm_processor_kwargs"] = mm_processor_kwargs
        return conversation, None, None

    monkeypatch.setattr(
        deepseek_v4_renderer,
        "parse_chat_messages",
        fake_parse_chat_messages,
    )

    class FakeRenderer:
        model_config = object()

        def _apply_chat_template(self, **kwargs):
            calls["template_kwargs"] = kwargs
            return "rendered"

    params = SimpleNamespace(
        media_io_kwargs=None,
        mm_processor_kwargs=None,
        get_apply_chat_template_kwargs=lambda: {},
    )
    rendered_conversation, prompt = DeepseekV4Renderer.render_messages(
        FakeRenderer(), messages, params
    )

    assert calls["content_format"] == "openai"
    assert calls["template_kwargs"]["conversation"] == conversation
    assert rendered_conversation == conversation
    assert prompt == {"prompt": "rendered"}


@pytest.mark.asyncio
async def test_deepseek_v4_async_renderer_keeps_structured_multimodal_content(
    monkeypatch,
):
    messages = [{"role": "user", "content": "hello"}]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "after"},
            ],
        }
    ]
    calls = {}

    async def fake_parse_chat_messages_async(
        messages_arg,
        model_config,
        content_format,
        media_io_kwargs=None,
        mm_processor_kwargs=None,
    ):
        calls["messages"] = messages_arg
        calls["model_config"] = model_config
        calls["content_format"] = content_format
        calls["media_io_kwargs"] = media_io_kwargs
        calls["mm_processor_kwargs"] = mm_processor_kwargs
        return conversation, None, None

    monkeypatch.setattr(
        deepseek_v4_renderer,
        "parse_chat_messages_async",
        fake_parse_chat_messages_async,
    )

    class FakeRenderer:
        model_config = object()

        async def _apply_chat_template_async(self, **kwargs):
            calls["template_kwargs"] = kwargs
            return "rendered"

    params = SimpleNamespace(
        media_io_kwargs=None,
        mm_processor_kwargs=None,
        get_apply_chat_template_kwargs=lambda: {},
    )
    rendered_conversation, prompt = await DeepseekV4Renderer.render_messages_async(
        FakeRenderer(), messages, params
    )

    assert calls["content_format"] == "openai"
    assert calls["template_kwargs"]["conversation"] == conversation
    assert rendered_conversation == conversation
    assert prompt == {"prompt": "rendered"}
