# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesCountTokensRequest,
    ResponsesCountTokensResponse,
)
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses


@pytest.mark.asyncio
async def test_count_responses_input_tokens():
    """
    Test that the count_tokens endpoint correctly sums prompt_token_ids
    from the engine inputs without running inference.
    """
    # 1. Create a mock instance of the serving class
    serving = MagicMock(spec=OpenAIServingResponses)
    serving._check_model = AsyncMock(return_value=None)
    serving.use_harmony = False
    serving.response_store = {}
    serving.response_store_lock = asyncio.Lock()
    
    # 2. Mock _make_request to return fake engine inputs with known token counts
    # We simulate a scenario where the input is split into two chunks
    # (e.g., context + new input)
    fake_engine_inputs = [
        {"prompt_token_ids": [1, 2, 3, 4, 5]},  # 5 tokens
        {"prompt_token_ids": [6, 7, 8]}          # 3 tokens
    ]
    serving._make_request = AsyncMock(return_value=(None, fake_engine_inputs))
    
    # 3. Create a valid request payload
    request = ResponsesCountTokensRequest(
        model="test-model",
        input="Hello world, this is a test!"
    )
    
    # 4. Call the method directly on the mock
    response = await OpenAIServingResponses.count_tokens(serving, request)
    
    # 5. Assertions
    assert isinstance(response, ResponsesCountTokensResponse)
    assert response.input_tokens == 8  # 5 + 3 tokens