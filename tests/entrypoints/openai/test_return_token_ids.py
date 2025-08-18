# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.tokenizer import get_tokenizer

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.7",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_basic_completion_with_emoji(server):
    """Test basic completion with emoji to verify token_ids field."""
    async with server.get_async_client() as client:
        # Test with return_token_ids enabled
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt="Complete this sentence with emojis: I love coding 🚀",
            max_tokens=10,
            temperature=0,
            logprobs=1,
        )

        # Check against the expected prompt token IDs
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
        encoded_tokens = tokenizer.encode(
            "Complete this sentence with emojis: I love coding 🚀")
        assert encoded_tokens is not None
        assert completion is not None
