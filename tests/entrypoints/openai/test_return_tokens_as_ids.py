# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Separate these tests out from test_completion and test_chat, because they
# require launching a second server with a different flag. Running both servers
# at the same time on a single node will OOM.

import pytest

from vllm.tokenizers import get_tokenizer

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def default_server_args(qwen3_lora_files):
    return [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        # lora config
        "--enable-lora",
        "--lora-modules",
        f"qwen3-lora={qwen3_lora_files}",
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
    ]


@pytest.fixture(scope="module")
def server_fixture(request, default_server_args):
    use_server_flag = request.param
    if use_server_flag:
        args_with_flag = default_server_args + ["--return-tokens-as-token-ids"]
        with RemoteOpenAIServer(MODEL_NAME, args_with_flag) as remote_server:
            yield (remote_server, True)
    else:
        with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
            yield (remote_server, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("server_fixture", [True, False], indirect=True)
async def test_completion_return_tokens_as_token_ids_completion(server_fixture):
    server, use_server_flag = server_fixture
    request_args = {}
    if not use_server_flag:
        request_args["return_tokens_as_token_ids"] = True

    async with server.get_async_client() as client:
        completion = await client.completions.create(
            model=MODEL_NAME,
            # Include Unicode characters to test for dividing a single
            # character across multiple tokens: 🎉 is [28705, 31862] for the
            # Zephyr tokenizer
            prompt="Say 'Hello, world! 🎉'",
            echo=True,
            temperature=0,
            max_tokens=10,
            logprobs=1,
            extra_body=request_args,
        )

        text = completion.choices[0].text
        token_strs = completion.choices[0].logprobs.tokens
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
        # Check that the token representations are consistent between raw
        # tokens and top_logprobs
        # Slice off the first one, because there's no scoring associated
        # with BOS
        top_logprobs = completion.choices[0].logprobs.top_logprobs[1:]
        top_logprob_keys = [
            next(iter(logprob_by_tokens)) for logprob_by_tokens in top_logprobs
        ]
        assert token_strs[1:] == top_logprob_keys

        # Check that decoding the tokens gives the expected text
        tokens = [int(token.removeprefix("token_id:")) for token in token_strs]
        assert text == tokenizer.decode(tokens, skip_special_tokens=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("server_fixture", [True, False], indirect=True)
async def test_chat_return_tokens_as_token_ids_completion(server_fixture):
    server, use_server_flag = server_fixture
    request_args = {}
    if not use_server_flag:
        request_args["return_tokens_as_token_ids"] = True

    async with server.get_async_client() as client:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            # Include Unicode characters to test for dividing a single
            # character across multiple tokens: 🎉 is [28705, 31862] for the
            # Zephyr tokenizer
            messages=[
                {
                    "role": "system",
                    "content": "You like to respond in only emojis, like 🎉",
                },
                {"role": "user", "content": "Please write some emojis: 🐱🐶🎉"},
            ],
            temperature=0,
            max_tokens=8,
            logprobs=True,
            extra_body=request_args,
        )

        text = response.choices[0].message.content
        tokenizer = get_tokenizer(tokenizer_name=MODEL_NAME)
        token_ids = []
        for logprob_content in response.choices[0].logprobs.content:
            token_ids.append(int(logprob_content.token.removeprefix("token_id:")))
        assert tokenizer.decode(token_ids, skip_special_tokens=True) == text
