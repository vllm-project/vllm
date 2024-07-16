import openai  # use the official client for correctness check
import pytest
import requests

from vllm.transformers_utils.tokenizer import get_tokenizer

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def server():
    with RemoteOpenAIServer([
            "--model",
            MODEL_NAME,
            # use half precision for speed and memory savings in CI environment
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "8192",
            "--enforce-eager",
            "--max-num-seqs",
            "128",
    ]) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server):
    return server.get_async_client()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_tokenize_completions(client: openai.AsyncOpenAI,
                                    model_name: str):
    base_url = str(client.base_url)[:-3].strip("/")
    tokenizer = get_tokenizer(tokenizer_name=model_name, tokenizer_mode="fast")

    for add_special in [False, True]:
        prompt = "This is a test prompt."
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
    "model_name",
    [MODEL_NAME],
)
async def test_tokenize_chat(client: openai.AsyncOpenAI, model_name: str):
    base_url = str(client.base_url)[:-3].strip("/")
    tokenizer = get_tokenizer(tokenizer_name=model_name, tokenizer_mode="fast")

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
                "content": "Can I ask a question?"
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
    "model_name",
    [MODEL_NAME],
)
async def test_detokenize(client: openai.AsyncOpenAI, model_name: str):
    base_url = str(client.base_url)[:-3].strip("/")
    tokenizer = get_tokenizer(tokenizer_name=model_name, tokenizer_mode="fast")

    prompt = "This is a test prompt."
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    response = requests.post(base_url + "/detokenize",
                             json={
                                 "model": model_name,
                                 "tokens": tokens
                             })
    response.raise_for_status()

    assert response.json() == {"prompt": prompt}
