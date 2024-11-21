"""Compare the embedding outputs of HF and vLLM models.

Run `pytest tests/models/embedding/language/test_embedding.py`.
"""
import math
import weakref

import pytest

from tests.conftest import HfRunner
from vllm import LLM
from vllm.distributed import (cleanup_dist_env_and_memory,
                              ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.utils import get_open_port

MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Bert
    "BAAI/bge-reranker-v2-m3",  # Roberta
]

TEXTS = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXT_PAIRS = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]


@pytest.fixture(scope="module", params=MODELS)
def model_name(request):
    yield request.param


@pytest.fixture(scope="module")
def llm(model_name):
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=model_name,
        max_num_batched_tokens=32768,
        #tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        dtype="half")

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)

        del llm

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def hf_model(model_name):
    yield HfRunner(model_name, dtype="half", is_cross_encoder=True)


@pytest.fixture
def distributed_init():
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0)
        ensure_model_parallel_initialized(1, 1)
    except Exception as e:
        print(e)


def test_llm_1_to_1(llm: LLM, hf_model, distributed_init):

    text_pair = [TEXTS[0], TEXT_PAIRS[0]]

    llm_output = llm.score(text_pair[0], text_pair[1])
    assert len(llm_output) == 1

    hf_output = hf_model.predict([text_pair]).tolist()
    assert len(hf_output) == 1

    assert math.isclose(hf_output[0],
                        llm_output[0].outputs.embedding[0],
                        rel_tol=0.01)


def test_llm_1_to_N(llm: LLM, hf_model, distributed_init):

    text_pairs = [
        [TEXTS[0], TEXT_PAIRS[0]],
        [TEXTS[0], TEXT_PAIRS[1]],
    ]

    llm_output = llm.score(TEXTS[0], TEXT_PAIRS)
    assert len(llm_output) == 2

    hf_output = hf_model.predict(text_pairs).tolist()
    assert len(hf_output) == 2

    assert math.isclose(hf_output[0],
                        llm_output[0].outputs.embedding[0],
                        rel_tol=0.01)
    assert math.isclose(hf_output[1],
                        llm_output[1].outputs.embedding[0],
                        rel_tol=0.01)


def test_llm_N_to_N(llm: LLM, hf_model, distributed_init):

    text_pairs = [
        [TEXTS[0], TEXT_PAIRS[0]],
        [TEXTS[1], TEXT_PAIRS[1]],
    ]

    llm_output = llm.score(TEXTS, TEXT_PAIRS)

    assert len(llm_output) == 2

    hf_output = hf_model.predict(text_pairs).tolist()
    assert len(hf_output) == 2

    assert math.isclose(hf_output[0],
                        llm_output[0].outputs.embedding[0],
                        rel_tol=0.01)
    assert math.isclose(hf_output[1],
                        llm_output[1].outputs.embedding[0],
                        rel_tol=0.01)
