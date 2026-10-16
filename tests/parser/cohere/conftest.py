# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from .utils import MockCohereTokenizer


@pytest.fixture(scope="package")
def tokenizer() -> MockCohereTokenizer:
    return MockCohereTokenizer()


@pytest.fixture
def request_obj() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], model="test-model")
