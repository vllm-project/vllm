import pytest
import requests

from vllm.entrypoints.openai.protocol import ScoreResponse

from ...utils import RemoteOpenAIServer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_text_1_str_text_2_list(server: RemoteOpenAIServer,
                                      model_name: str):
    text_1 = "What is the capital of France?"
    text_2 = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    score_response = requests.post(server.url_for("score"),
                                   json={
                                       "model": model_name,
                                       "text_1": text_1,
                                       "text_2": text_2,
                                   })
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 2
    assert score.data[0].score <= 0.01
    assert score.data[1].score >= 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_text_1_list_text_2_list(server: RemoteOpenAIServer,
                                       model_name: str):
    text_1 = [
        "What is the capital of the United States?",
        "What is the capital of France?"
    ]
    text_2 = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]

    score_response = requests.post(server.url_for("score"),
                                   json={
                                       "model": model_name,
                                       "text_1": text_1,
                                       "text_2": text_2,
                                   })
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 2
    assert score.data[0].score <= 0.01
    assert score.data[1].score >= 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_text_1_str_text_2_str(server: RemoteOpenAIServer,
                                     model_name: str):
    text_1 = "What is the capital of France?"
    text_2 = "The capital of France is Paris."

    score_response = requests.post(server.url_for("score"),
                                   json={
                                       "model": model_name,
                                       "text_1": text_1,
                                       "text_2": text_2,
                                   })
    score_response.raise_for_status()
    score = ScoreResponse.model_validate(score_response.json())

    assert score.id is not None
    assert score.data is not None
    assert len(score.data) == 1
    assert score.data[0].score >= 0.9
