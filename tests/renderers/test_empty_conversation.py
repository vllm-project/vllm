# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An empty conversation must be rejected whatever the renderer.

The HF tokenizer refuses one inside ``apply_chat_template``, but renderers with
their own chat encoding (e.g. DeepSeek V3.2/V4/V4.1) render it to a bare BOS
prompt, from which the model generates unrelated text.
"""

from types import SimpleNamespace

import pytest
from openai_harmony import Message, Role

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.exceptions import VLLMValidationError
from vllm.renderers.base import BaseRenderer
from vllm.renderers.online_renderer import OnlineRenderer
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


@pytest.fixture
def harmony_renderer(monkeypatch: pytest.MonkeyPatch) -> OnlineRenderer:
    renderer = OnlineRenderer.__new__(OnlineRenderer)
    renderer.use_harmony = True
    renderer.parser = None
    renderer.trust_request_chat_template = True
    renderer.exclude_tools_when_tool_choice_none = False
    renderer.reasoning_effort_map = {}
    renderer.supports_browsing = False
    renderer.supports_code_interpreter = False
    renderer.model_config = SimpleNamespace(max_model_len=100)
    renderer.renderer = SimpleNamespace(
        tokenizer=SimpleNamespace(truncation_side="left")
    )
    monkeypatch.setattr(
        "vllm.renderers.online_renderer.render_for_completion", lambda messages: [1]
    )
    return renderer


def test_harmony_chat_rejects_an_empty_conversation(harmony_renderer):
    request = ChatCompletionRequest(model="gpt-oss", messages=[])
    with pytest.raises(VLLMValidationError, match="empty conversation"):
        harmony_renderer._make_request_with_harmony(request)


def test_harmony_chat_renders_a_non_empty_conversation(harmony_renderer):
    request = ChatCompletionRequest(
        model="gpt-oss", messages=[{"role": "user", "content": "hi"}]
    )
    _, (engine_input,) = harmony_renderer._make_request_with_harmony(request)
    assert engine_input["prompt_token_ids"] == [1]


@pytest.mark.asyncio
async def test_harmony_responses_rejects_an_empty_conversation(harmony_renderer):
    result = await harmony_renderer.render_responses(ResponsesRequest(input=[]))
    assert isinstance(result, ErrorResponse)
    assert "empty conversation" in result.error.message
    assert result.error.param == "input"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_kwargs", "previous_messages"),
    [
        ({"instructions": "Be brief."}, None),
        ({}, [Message.from_role_and_content(Role.USER, "first")]),
    ],
    ids=["instructions", "previous_response"],
)
async def test_harmony_responses_renders_empty_input_with_context(
    harmony_renderer, request_kwargs, previous_messages
):
    result = await harmony_renderer.render_responses(
        ResponsesRequest(input=[], **request_kwargs),
        previous_messages=previous_messages,
    )
    assert not isinstance(result, ErrorResponse), result
    assert result.engine_input["prompt_token_ids"] == [1]
