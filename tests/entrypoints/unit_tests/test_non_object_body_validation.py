# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Non-object JSON bodies must fail validation cleanly (4xx), not AttributeError (500).

mode=before validators that call data.get(...) without an isinstance(data, dict)
guard raise AttributeError for string/list/scalar bodies and surface as HTTP 500.

This extends the chat completion coverage added in #51654 to the remaining
request models whose before-validators were missing the same guard.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.pooling.classify.protocol import ClassificationChatRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingChatRequest
from vllm.entrypoints.pooling.pooling.protocol import PoolingChatRequest
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from vllm.exceptions import VLLMValidationError

pytestmark = pytest.mark.skip_global_cleanup

REQUEST_MODELS = [
    CompletionRequest,
    ResponsesRequest,
    EmbeddingChatRequest,
    ClassificationChatRequest,
    PoolingChatRequest,
    TokenizeChatRequest,
    TranscriptionRequest,
    TranslationRequest,
]


@pytest.mark.parametrize("request_model", REQUEST_MODELS, ids=lambda m: m.__name__)
@pytest.mark.parametrize(
    "payload",
    [
        "this is not valid json{{{",
        ["not", "an", "object"],
        42,
        None,
        True,
    ],
)
def test_request_models_reject_non_object_body(request_model, payload):
    with pytest.raises(ValidationError):
        request_model.model_validate(payload)


def test_completion_request_still_validates_dict_bodies():
    """The guard must not swallow real field-level errors on object bodies."""
    with pytest.raises(VLLMValidationError, match="prompt"):
        CompletionRequest.model_validate({"model": "qwen", "prompt": ""})


def test_tokenize_chat_request_still_validates_dict_bodies():
    with pytest.raises(VLLMValidationError, match="add_generation_prompt"):
        TokenizeChatRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "continue_final_message": True,
                "add_generation_prompt": True,
            }
        )
