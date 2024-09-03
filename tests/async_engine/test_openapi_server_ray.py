import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from ..utils import VLLM_PATH, RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "facebook/opt-125m"
chatml_jinja_path = VLLM_PATH / "examples/template_chatml.jinja"
assert chatml_jinja_path.exists()


@pytest.fixture(scope="module")
def server():
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--engine-use-ray",
        "--chat-template",
        str(chatml_jinja_path),
    ]

    # Allow `--engine-use-ray`, otherwise the launch of the server throw
    # an error due to try to use a deprecated feature
    env_dict = {"VLLM_ALLOW_ENGINE_USE_RAY": "1"}
    with RemoteOpenAIServer(MODEL_NAME, args,
                            env_dict=env_dict) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_check_models(client: openai.AsyncOpenAI):
    models = await client.models.list()
    models = models.data
    served_model = models[0]
    assert served_model.id == MODEL_NAME
    assert all(model.root == MODEL_NAME for model in models)


@pytest.mark.asyncio
async def test_single_completion(client: openai.AsyncOpenAI):
    completion = await client.completions.create(model=MODEL_NAME,
                                                 prompt="Hello, my name is",
                                                 max_tokens=5,
                                                 temperature=0.0)

    assert completion.id is not None
    assert len(completion.choices) == 1
    assert len(completion.choices[0].text) >= 5
    assert completion.choices[0].finish_reason == "length"
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=6, total_tokens=11)

    # test using token IDs
    completion = await client.completions.create(
        model=MODEL_NAME,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 5


@pytest.mark.asyncio
async def test_single_chat_session(client: openai.AsyncOpenAI):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(model=MODEL_NAME,
                                                           messages=messages,
                                                           max_tokens=10,
                                                           logprobs=True,
                                                           top_logprobs=5)
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=10, prompt_tokens=55, total_tokens=65)

    message = choice.message
    assert message.content is not None and len(message.content) >= 10
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0
