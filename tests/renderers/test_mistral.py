# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig
from vllm.renderers.mistral import MistralRenderer
from vllm.tokenizers.mistral import MistralTokenizer


@pytest.mark.asyncio
async def test_async_mistral_tokenizer_does_not_block_event_loop():
    expected_tokens = [1, 2, 3]

    # Mock the blocking version to sleep
    def mocked_apply_chat_template(*_args, **_kwargs):
        time.sleep(2)
        return expected_tokens

    mock_tokenizer = Mock(spec=MistralTokenizer)
    mock_tokenizer.apply_chat_template = mocked_apply_chat_template
    mock_renderer = MistralRenderer(Mock(spec=ModelConfig), tokenizer_kwargs={})
    mock_renderer._tokenizer = mock_tokenizer

    task = mock_renderer.render_messages_async([])

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
    _, prompt = await task
    assert prompt["prompt_token_ids"] == expected_tokens, (
        "Mocked blocking tokenizer was not called"
    )
    assert blocked_count == 0, "Event loop blocked during tokenization"
