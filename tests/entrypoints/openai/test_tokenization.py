# SPDX-License-Identifier: Apache-2.0

import pytest
import pytest_asyncio
import requests

from vllm.transformers_utils.tokenizer import get_tokenizer

from ...utils import RemoteOpenAIServer
from .test_completion import zephyr_lora_added_tokens_files  # noqa: F401
from .test_completion import zephyr_lora_files  # noqa: F401

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def server(zephyr_lora_added_tokens_files: str):  # noqa: F811
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
        # lora config
        "--enable-lora",
        "--lora-modules",
        f"zephyr-lora2={zephyr_lora_added_tokens_files}",
        "--max-lora-rank",
        "64",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def tokenizer_name(model_name: str,
                   zephyr_lora_added_tokens_files: str):  # noqa: F811
    return zephyr_lora_added_tokens_files if (
        model_name == "zephyr-lora2") else model_name


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_completions(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    for add_special in [False, True]:
        prompt = "vllm1 This is a test prompt."
        tokens = tokenizer.encode(prompt, add_special_tokens=add_special)

        response = requests.post(server.url_for("tokenize"),
                                 json={
                                     "add_special_tokens": add_special,
                                     "model": model_name,
                                     "prompt": prompt
                                 })
        response.raise_for_status()

        result = response.json()
        assert result["tokens"] == tokens
        assert result["count"] == len(tokens)
        assert result["max_model_len"] == 8192
        assert result["token_strs"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_chat(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    for add_generation in [False, True]:
        for add_special in [False, True]:
            conversation = [{
                "role": "user",
                "content": "Hi there!"
            }, {
                "role": "assistant",
                "content": "Nice to meet you!"
            }, {
                "role": "user",
                "content": "Can I ask a question? vllm1"
            }]
            for continue_final in [False, True]:
                if add_generation and continue_final:
                    continue
                if continue_final:
                    conversation.append({
                        "role": "assistant",
                        "content": "Sure,"
                    })

                prompt = tokenizer.apply_chat_template(
                    add_generation_prompt=add_generation,
                    continue_final_message=continue_final,
                    conversation=conversation,
                    tokenize=False)
                tokens = tokenizer.encode(prompt,
                                          add_special_tokens=add_special)

                response = requests.post(server.url_for("tokenize"),
                                         json={
                                             "add_generation_prompt":
                                             add_generation,
                                             "continue_final_message":
                                             continue_final,
                                             "add_special_tokens": add_special,
                                             "messages": conversation,
                                             "model": model_name
                                         })
                response.raise_for_status()

                result = response.json()
                assert result["tokens"] == tokens
                assert result["count"] == len(tokens)
                assert result["max_model_len"] == 8192
                assert result["token_strs"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_chat_with_tools(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    for add_generation in [False, True]:
        for add_special in [False, True]:
            conversation = [{
                "role":
                "user",
                "content":
                "What's the weather like in Paris today?",
            }]

            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string"
                            }
                        },
                    },
                },
            }]

            for continue_final in [False, True]:
                if add_generation and continue_final:
                    continue
                if continue_final:
                    conversation.append({
                        "role": "assistant",
                        "content": "Sure,"
                    })

                prompt = tokenizer.apply_chat_template(
                    add_generation_prompt=add_generation,
                    continue_final_message=continue_final,
                    conversation=conversation,
                    tools=tools,
                    tokenize=False,
                )
                tokens = tokenizer.encode(prompt,
                                          add_special_tokens=add_special)

                response = requests.post(
                    server.url_for("tokenize"),
                    json={
                        "add_generation_prompt": add_generation,
                        "continue_final_message": continue_final,
                        "add_special_tokens": add_special,
                        "messages": conversation,
                        "model": model_name,
                        "tools": tools,
                    },
                )
                response.raise_for_status()

                result = response.json()
                assert result["tokens"] == tokens
                assert result["count"] == len(tokens)
                assert result["max_model_len"] == 8192
                assert result["token_strs"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name, tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_with_return_token_strs(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    prompt = "This is a token_strs test prompt! vllm1"
    response = requests.post(
        server.url_for("tokenize"),
        json={
            "prompt": prompt,
            "model": model_name,
            "return_token_strs": True
        },
    )
    response.raise_for_status()

    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    tokens_str = tokenizer.convert_ids_to_tokens(tokens)

    result = response.json()
    assert result["tokens"] == tokens
    assert result["count"] == len(tokens)
    assert result["max_model_len"] == 8192
    assert result["token_strs"] == tokens_str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_detokenize(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    prompt = "This is a test prompt. vllm1"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    response = requests.post(server.url_for("detokenize"),
                             json={
                                 "model": model_name,
                                 "tokens": tokens
                             })
    response.raise_for_status()

    assert response.json() == {"prompt": prompt}
