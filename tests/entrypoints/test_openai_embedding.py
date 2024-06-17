import openai
import pytest
import ray

from ..utils import VLLM_PATH, RemoteOpenAIServer

EMBEDDING_MODEL_NAME = "intfloat/e5-mistral-7b-instruct"

pytestmark = pytest.mark.openai


@pytest.fixture(scope="module")
def ray_ctx():
    ray.init(runtime_env={"working_dir": VLLM_PATH})
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def embedding_server(ray_ctx):
    return RemoteOpenAIServer([
        "--model",
        EMBEDDING_MODEL_NAME,
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-model-len",
        "8192",
        "--enforce-eager",
    ])


@pytest.mark.asyncio
@pytest.fixture(scope="module")
def embedding_client(embedding_server):
    return embedding_server.get_async_client()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [EMBEDDING_MODEL_NAME],
)
async def test_single_embedding(embedding_client: openai.AsyncOpenAI,
                                model_name: str):
    input_texts = [
        "The chef prepared a delicious meal.",
    ]

    # test single embedding
    embeddings = await embedding_client.embeddings.create(
        model=model_name,
        input=input_texts,
        encoding_format="float",
    )
    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 4096
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 9
    assert embeddings.usage.total_tokens == 9

    # test using token IDs
    input_tokens = [1, 1, 1, 1, 1]
    embeddings = await embedding_client.embeddings.create(
        model=model_name,
        input=input_tokens,
        encoding_format="float",
    )
    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 4096
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 5
    assert embeddings.usage.total_tokens == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [EMBEDDING_MODEL_NAME],
)
async def test_batch_embedding(embedding_client: openai.AsyncOpenAI,
                               model_name: str):
    # test List[str]
    input_texts = [
        "The cat sat on the mat.", "A feline was resting on a rug.",
        "Stars twinkle brightly in the night sky."
    ]
    embeddings = await embedding_client.embeddings.create(
        model=model_name,
        input=input_texts,
        encoding_format="float",
    )
    assert embeddings.id is not None
    assert len(embeddings.data) == 3
    assert len(embeddings.data[0].embedding) == 4096

    # test List[List[int]]
    input_tokens = [[4, 5, 7, 9, 20], [15, 29, 499], [24, 24, 24, 24, 24],
                    [25, 32, 64, 77]]
    embeddings = await embedding_client.embeddings.create(
        model=model_name,
        input=input_tokens,
        encoding_format="float",
    )
    assert embeddings.id is not None
    assert len(embeddings.data) == 4
    assert len(embeddings.data[0].embedding) == 4096
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == 17
    assert embeddings.usage.total_tokens == 17
