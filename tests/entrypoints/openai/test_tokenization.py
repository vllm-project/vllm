import openai  # use the official client for correctness check
import pytest
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


@pytest.fixture(scope="module")
def client(server):
    return server.get_async_client()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_completions(client: openai.AsyncOpenAI,
                                    model_name: str, tokenizer_name: str):
    base_url = str(client.base_url)[:-3].strip("/")
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    for add_special in [False, True]:
        prompt = "vllm1 This is a test prompt."
        tokens = tokenizer.encode(prompt, add_special_tokens=add_special)

        response = requests.post(base_url + "/tokenize",
                                 json={
                                     "add_special_tokens": add_special,
                                     "model": model_name,
                                     "prompt": prompt
                                 })
        response.raise_for_status()

        assert response.json() == {
            "tokens": tokens,
            "count": len(tokens),
            "max_model_len": 8192
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_tokenize_chat(client: openai.AsyncOpenAI, model_name: str,
                             tokenizer_name: str):
    base_url = str(client.base_url)[:-3].strip("/")
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

            prompt = tokenizer.apply_chat_template(
                add_generation_prompt=add_generation,
                conversation=conversation,
                tokenize=False)
            tokens = tokenizer.encode(prompt, add_special_tokens=add_special)

            response = requests.post(base_url + "/tokenize",
                                     json={
                                         "add_generation_prompt":
                                         add_generation,
                                         "add_special_tokens": add_special,
                                         "messages": conversation,
                                         "model": model_name
                                     })
            response.raise_for_status()

            assert response.json() == {
                "tokens": tokens,
                "count": len(tokens),
                "max_model_len": 8192
            }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenizer_name",
    [(MODEL_NAME, MODEL_NAME), ("zephyr-lora2", "zephyr-lora2")],
    indirect=["tokenizer_name"],
)
async def test_detokenize(client: openai.AsyncOpenAI, model_name: str,
                          tokenizer_name: str):
    base_url = str(client.base_url)[:-3].strip("/")
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                              tokenizer_mode="fast")

    prompt = "This is a test prompt. vllm1"
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    print(f"CALLING {base_url} FOR {model_name}")
    response = requests.post(base_url + "/detokenize",
                             json={
                                 "model": model_name,
                                 "tokens": tokens
                             })
    response.raise_for_status()

    assert response.json() == {"prompt": prompt}
