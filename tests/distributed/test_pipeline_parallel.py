import os

import openai  # use the official client for correctness check
import pytest
# using Ray for overall ease of process management, parallel requests,
# and debugging.
import ray

from ..utils import VLLM_PATH, RemoteOpenAIServer

# downloading lora to test lora requests

# any model with a chat template should work here
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
EAGER_MODE = bool(int(os.getenv("EAGER_MODE", 0)))
CHUNKED_PREFILL = bool(int(os.getenv("CHUNKED_PREFILL", 0)))
TP_SIZE = int(os.getenv("TP_SIZE", 1))
PP_SIZE = int(os.getenv("PP_SIZE", 1))

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def ray_ctx():
    ray.init(runtime_env={"working_dir": VLLM_PATH})
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def server(ray_ctx):
    args = [
        "--model",
        MODEL_NAME,
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--distributed-executor-backend",
        "ray",
    ]
    if CHUNKED_PREFILL:
        args += [
            "--enable-chunked-prefill",
        ]
    if EAGER_MODE:
        args += [
            "--enforce-eager",
        ]
    return RemoteOpenAIServer(args, num_gpus=PP_SIZE * TP_SIZE)


@pytest.fixture(scope="module")
def client(server):
    return server.get_async_client()


async def test_check_models(server, client: openai.AsyncOpenAI):
    models = await client.models.list()
    models = models.data
    served_model = models[0]
    assert served_model.id == MODEL_NAME
    assert all(model.root == MODEL_NAME for model in models)


@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_single_completion(server, client: openai.AsyncOpenAI,
                                 model_name: str):
    completion = await client.completions.create(model=model_name,
                                                 prompt="Hello, my name is",
                                                 max_tokens=5,
                                                 temperature=0.0)

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1
    assert completion.choices[0].text is not None and len(
        completion.choices[0].text) >= 5
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
    assert completion.choices[0].text is not None and len(
        completion.choices[0].text) >= 5


@pytest.mark.parametrize(
    # just test 1 lora hereafter
    "model_name",
    [MODEL_NAME],
)
async def test_batch_completions(server, client: openai.AsyncOpenAI,
                                 model_name: str):
    # test simple list
    batch = await client.completions.create(
        model=model_name,
        prompt=["Hello, my name is", "Hello, my name is"],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(batch.choices) == 2
    assert batch.choices[0].text == batch.choices[1].text

    # test n = 2
    batch = await client.completions.create(
        model=model_name,
        prompt=["Hello, my name is", "Hello, my name is"],
        n=2,
        max_tokens=5,
        temperature=0.0,
        extra_body=dict(
            # NOTE: this has to be true for n > 1 in vLLM, but not necessary
            # for official client.
            use_beam_search=True),
    )
    assert len(batch.choices) == 4
    assert batch.choices[0].text != batch.choices[
        1].text, "beam search should be different"
    assert batch.choices[0].text == batch.choices[
        2].text, "two copies of the same prompt should be the same"
    assert batch.choices[1].text == batch.choices[
        3].text, "two copies of the same prompt should be the same"

    # test streaming
    batch = await client.completions.create(
        model=model_name,
        prompt=["Hello, my name is", "Hello, my name is"],
        max_tokens=5,
        temperature=0.0,
        stream=True,
    )
    texts = [""] * 2
    async for chunk in batch:
        assert len(chunk.choices) == 1
        choice = chunk.choices[0]
        texts[choice.index] += choice.text
    assert texts[0] == texts[1]
