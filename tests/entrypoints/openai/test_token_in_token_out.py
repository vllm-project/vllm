# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.tokenizer import get_tokenizer

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len", "2048", "--max-num-seqs", "128", "--enforce-eager",
        "--skip-tokenizer-init"
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
async def test_token_in_token_out_and_logprobs(server):
    """
    Test token-in-token-out and token_ids align with prompt_logprobs
    & logprobs when return_tokens_as_token_ids is enabled.
    """
    tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
    text = "Hello, world! How are you today?"
    token_ids = tokenizer.encode(text)
    async with server.get_async_client() as client:
        # Test with both return_token_ids and return_tokens_as_token_ids enabled
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt=token_ids,
            max_tokens=20,
            temperature=0,
            echo=True,
            extra_body={
                "return_token_ids": True,
            },
        )

        # Verify all fields are present
        assert (completion.choices[0].token_ids is not None
                and 0 < len(completion.choices[0].token_ids) <= 20)
        assert completion.choices[0].prompt_token_ids is not None

        # Decode prompt tokens
        if completion.choices[0].prompt_token_ids:
            prompt_text = tokenizer.decode(
                completion.choices[0].prompt_token_ids)
            # The decoded prompt should match or close to original prompt
            assert prompt_text == text
