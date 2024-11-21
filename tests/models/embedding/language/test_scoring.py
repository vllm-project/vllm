"""Compare the embedding outputs of HF and vLLM models.

Run `pytest tests/models/embedding/language/test_embedding.py`.
"""
import math

import pytest

MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Bert
    "BAAI/bge-reranker-v2-m3",  # Roberta
]

TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]


@pytest.fixture(scope="module", params=MODELS)
def model_name(request):
    yield request.param


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_1_to_1(vllm_runner, hf_runner, model_name, dtype: str):

    text_pair = [TEXTS_1[0], TEXTS_2[0]]

    with hf_runner(model_name, dtype=dtype, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict([text_pair]).tolist()

    with vllm_runner(model_name,
                     task="embedding",
                     dtype=dtype,
                     max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(text_pair[0], text_pair[1])

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1

    assert math.isclose(hf_outputs[0], vllm_outputs[0][0], rel_tol=0.01)


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_1_to_N(vllm_runner, hf_runner, model_name, dtype: str):

    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]

    with hf_runner(model_name, dtype=dtype, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict(text_pairs).tolist()

    with vllm_runner(model_name,
                     task="embedding",
                     dtype=dtype,
                     max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1[0], TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert math.isclose(hf_outputs[0], vllm_outputs[0][0], rel_tol=0.01)
    assert math.isclose(hf_outputs[1], vllm_outputs[1][0], rel_tol=0.01)


@pytest.mark.parametrize("dtype", ["half"])
def test_llm_N_to_N(vllm_runner, hf_runner, model_name, dtype: str):

    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]

    with hf_runner(model_name, dtype=dtype, is_cross_encoder=True) as hf_model:
        hf_outputs = hf_model.predict(text_pairs).tolist()

    with vllm_runner(model_name,
                     task="embedding",
                     dtype=dtype,
                     max_model_len=None) as vllm_model:
        vllm_outputs = vllm_model.score(TEXTS_1, TEXTS_2)

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert math.isclose(hf_outputs[0], vllm_outputs[0][0], rel_tol=0.01)
    assert math.isclose(hf_outputs[1], vllm_outputs[1][0], rel_tol=0.01)
