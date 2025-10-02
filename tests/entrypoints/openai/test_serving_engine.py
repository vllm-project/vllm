# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from vllm.config import ModelConfig
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels


@pytest.fixture()
def serving() -> OpenAIServing:
    """Create a minimal OpenAIServing instance for testing."""

    # Create minimal mocks
    engine_client = Mock()
    model_config = Mock(spec=ModelConfig)
    model_config.max_model_len = 32768
    models = Mock(spec=OpenAIServingModels)

    serving = OpenAIServing(
        engine_client=engine_client,
        model_config=model_config,
        models=models,
        request_logger=None,
    )
    return serving


@pytest.fixture()
def large_user_message() -> dict[str, str]:
    words_needed = 200_000
    content = " ".join([f"word{i+1}" for i in range(words_needed)])
    return {"role": "user", "content": content}


@pytest.mark.asyncio
@patch('vllm.entrypoints.openai.serving_engine.apply_mistral_chat_template')
async def test_async_mistral_tokenizer_does_not_block_event_loop(
        mock_apply_mistral_chat_template, serving: OpenAIServing,
        large_user_message: dict[str, str]):
    expected_tokens = [1, 2, 3]

    # Mock the blocking version to sleep
    def mock_tokenizer(*args, **kwargs):
        time.sleep(2)
        return expected_tokens

    mock_apply_mistral_chat_template.side_effect = mock_tokenizer

    task = asyncio.create_task(
        serving._async_apply_mistral_chat_template(None, [large_user_message],
                                                   chat_template=None,
                                                   tools=None))

    # Ensure the event loop is not blocked
    blocked_count = 0
    for _i in range(20):  # Check over ~2 seconds
        start = time.perf_counter()
        await asyncio.sleep(0)
        elapsed = time.perf_counter() - start

        # an overly generous elapsed time for slow machines
        if elapsed >= 0.5:
            blocked_count += 1

        await asyncio.sleep(0.1)

    # Ensure task completes
    tokens = await task
    assert tokens == expected_tokens, "Mocked blocking tokenizer was not called"
    assert blocked_count == 0, ("Event loop blocked during tokenization")
