# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Falsy-but-explicit request values must not be read as "field omitted".

Each case below covers a request field typed ``X | None`` whose consumer used a
truthiness test, so a caller-supplied falsy value took the branch reserved for
an omitted field.

This extends the falsy-field coverage of #48098 to the chat template and stream
option paths.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingChatRequest
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
from vllm.exceptions import VLLMValidationError

pytestmark = pytest.mark.skip_global_cleanup

SERVER_TEMPLATE = "{{ 'server default' }}"

CHAT_BODY = {"model": "qwen", "messages": [{"role": "user", "content": "hello"}]}

CHAT_TEMPLATE_MODELS = [
    ChatCompletionRequest,
    TokenizeChatRequest,
    EmbeddingChatRequest,
]

STREAM_OPTION_BODIES = [
    (ChatCompletionRequest, CHAT_BODY),
    (CompletionRequest, {"model": "qwen", "prompt": "hello"}),
]
IDS = [model.__name__ for model, _ in STREAM_OPTION_BODIES]


@pytest.mark.parametrize(
    "request_model", CHAT_TEMPLATE_MODELS, ids=lambda m: m.__name__
)
def test_empty_chat_template_is_not_replaced_by_the_server_default(request_model):
    """``chat_template: ""`` must reach the renderer as the caller wrote it.

    ``OnlineRenderer.validate_chat_template`` already counts ``""`` as a
    caller-supplied template and refuses it unless
    ``--trust-request-chat-template`` is set, so substituting the server
    template afterwards renders a prompt the caller never asked for.
    """
    request = request_model.model_validate({**CHAT_BODY, "chat_template": ""})

    chat_params = request.build_chat_params(SERVER_TEMPLATE, "auto")

    assert chat_params.chat_template == ""


@pytest.mark.parametrize(
    "request_model", CHAT_TEMPLATE_MODELS, ids=lambda m: m.__name__
)
def test_omitted_chat_template_still_falls_back_to_the_server_default(request_model):
    request = request_model.model_validate(dict(CHAT_BODY))

    chat_params = request.build_chat_params(SERVER_TEMPLATE, "auto")

    assert chat_params.chat_template == SERVER_TEMPLATE


@pytest.mark.parametrize(("request_model", "body"), STREAM_OPTION_BODIES, ids=IDS)
def test_empty_stream_options_are_refused_without_streaming(request_model, body):
    """``stream_options: {}`` is still ``stream_options``.

    The field is meaningless outside a streaming request, and an empty object
    is no more meaningful there than a populated one.
    """
    with pytest.raises(VLLMValidationError, match="stream_options"):
        request_model.model_validate({**body, "stream": False, "stream_options": {}})


@pytest.mark.parametrize(("request_model", "body"), STREAM_OPTION_BODIES, ids=IDS)
def test_empty_stream_options_are_accepted_with_streaming(request_model, body):
    request = request_model.model_validate(
        {**body, "stream": True, "stream_options": {}}
    )

    assert request.stream_options is not None


@pytest.mark.parametrize(("request_model", "body"), STREAM_OPTION_BODIES, ids=IDS)
def test_omitted_stream_options_are_accepted_without_streaming(request_model, body):
    request = request_model.model_validate({**body, "stream": False})

    assert request.stream_options is None
