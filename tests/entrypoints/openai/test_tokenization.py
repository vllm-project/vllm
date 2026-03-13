# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import pytest_asyncio
import requests

from vllm.tokenizers import get_tokenizer

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def server():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
        "--enable-tokenizer-info-endpoint",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def tokenizer_name(model_name: str):
    return model_name


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME)],
    indirect=["tokenizer_name"],
)
async def test_tokenize_completions(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name)

    for add_special in [False, True]:
        prompt = "vllm1 This is a test prompt."
        tokens = tokenizer.encode(prompt, add_special_tokens=add_special)

        response = requests.post(
            server.url_for("tokenize"),
            json={
                "add_special_tokens": add_special,
                "model": model_name,
                "prompt": prompt,
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
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME)],
    indirect=["tokenizer_name"],
)
async def test_tokenize_chat(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name)

    for add_generation in [False, True]:
        for add_special in [False, True]:
            conversation = [
                {"role": "user", "content": "Hi there!"},
                {"role": "assistant", "content": "Nice to meet you!"},
                {"role": "user", "content": "Can I ask a question? vllm1"},
            ]
            for continue_final in [False, True]:
                if add_generation and continue_final:
                    continue
                if continue_final:
                    conversation.append({"role": "assistant", "content": "Sure,"})

                prompt = tokenizer.apply_chat_template(
                    add_generation_prompt=add_generation,
                    continue_final_message=continue_final,
                    conversation=conversation,
                    tokenize=False,
                )
                tokens = tokenizer.encode(prompt, add_special_tokens=add_special)

                response = requests.post(
                    server.url_for("tokenize"),
                    json={
                        "add_generation_prompt": add_generation,
                        "continue_final_message": continue_final,
                        "add_special_tokens": add_special,
                        "messages": conversation,
                        "model": model_name,
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
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME)],
    indirect=["tokenizer_name"],
)
async def test_tokenize_chat_with_tools(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name)

    for add_generation in [False, True]:
        for add_special in [False, True]:
            conversation = [
                {
                    "role": "user",
                    "content": "What's the weather like in Paris today?",
                }
            ]

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ]

            for continue_final in [False, True]:
                if add_generation and continue_final:
                    continue
                if continue_final:
                    conversation.append({"role": "assistant", "content": "Sure,"})

                prompt = tokenizer.apply_chat_template(
                    add_generation_prompt=add_generation,
                    continue_final_message=continue_final,
                    conversation=conversation,
                    tools=tools,
                    tokenize=False,
                )
                tokens = tokenizer.encode(prompt, add_special_tokens=add_special)

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
    [(MODEL_NAME, MODEL_NAME)],
    indirect=["tokenizer_name"],
)
async def test_tokenize_with_return_token_strs(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name)

    prompt = "This is a token_strs test prompt! vllm1"
    response = requests.post(
        server.url_for("tokenize"),
        json={"prompt": prompt, "model": model_name, "return_token_strs": True},
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
    [(MODEL_NAME, MODEL_NAME)],
    indirect=["tokenizer_name"],
)
async def test_detokenize(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name)

    prompt = "This is a test prompt. vllm1"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    response = requests.post(
        server.url_for("detokenize"), json={"model": model_name, "tokens": tokens}
    )
    response.raise_for_status()

    assert response.json() == {"prompt": prompt}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME)],
    indirect=["tokenizer_name"],
)
async def test_tokenizer_info_basic(
    server: RemoteOpenAIServer,
    model_name: str,
    tokenizer_name: str,
):
    """Test basic tokenizer info endpoint functionality."""
    response = requests.get(server.url_for("tokenizer_info"))
    response.raise_for_status()
    result = response.json()
    assert "tokenizer_class" in result
    assert isinstance(result["tokenizer_class"], str)
    assert result["tokenizer_class"]


@pytest.mark.asyncio
async def test_tokenizer_info_schema(server: RemoteOpenAIServer):
    """Test that the response matches expected schema types."""
    response = requests.get(server.url_for("tokenizer_info"))
    response.raise_for_status()
    result = response.json()
    field_types = {
        "add_bos_token": bool,
        "add_prefix_space": bool,
        "clean_up_tokenization_spaces": bool,
        "split_special_tokens": bool,
        "bos_token": str,
        "eos_token": str,
        "pad_token": str,
        "unk_token": str,
        "chat_template": str,
        "errors": str,
        "model_max_length": int,
        "additional_special_tokens": list,
        "added_tokens_decoder": dict,
    }
    for field, expected_type in field_types.items():
        if field in result and result[field] is not None:
            assert isinstance(result[field], expected_type), (
                f"{field} should be {expected_type.__name__}"
            )


@pytest.mark.asyncio
async def test_tokenizer_info_consistency_with_tokenize(
    server: RemoteOpenAIServer,
):
    """Test that tokenizer info is consistent with tokenization endpoint."""
    info_response = requests.get(server.url_for("tokenizer_info"))
    info_response.raise_for_status()
    info = info_response.json()
    tokenize_response = requests.post(
        server.url_for("tokenize"),
        json={"model": MODEL_NAME, "prompt": "Hello world!"},
    )
    tokenize_response.raise_for_status()
    tokenize_result = tokenize_response.json()
    info_max_len = info.get("model_max_length")
    tokenize_max_len = tokenize_result.get("max_model_len")
    if info_max_len and tokenize_max_len:
        assert info_max_len >= tokenize_max_len, (
            "Info max length should be >= tokenize max length"
        )


@pytest.mark.asyncio
async def test_tokenizer_info_chat_template(server: RemoteOpenAIServer):
    """Test chat template is properly included."""
    response = requests.get(server.url_for("tokenizer_info"))
    response.raise_for_status()
    result = response.json()
    chat_template = result.get("chat_template")
    if chat_template:
        assert isinstance(chat_template, str), "Chat template should be a string"
        assert chat_template.strip(), "Chat template should not be empty"
