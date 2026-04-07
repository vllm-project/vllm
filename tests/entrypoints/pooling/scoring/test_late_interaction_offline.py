# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

from .util import ColBERTScoringHfRunner

MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
COLBERT_DIM = 96

LINEAR_WEIGHTS_KEY = "linear.weight"
PROMPT = "The chef prepared a delicious meal."

TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]

DTYPE = "half"


@pytest.fixture(scope="module")
def llm():
    # ROCm: Use FLEX_ATTENTION backend as it's the only attention backend
    # that supports encoder-only models on ROCm.
    attention_config = None
    if current_platform.is_rocm():
        attention_config = {"backend": "FLEX_ATTENTION"}

    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
        attention_config=attention_config,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def hf_model():
    return ColBERTScoringHfRunner(
        model_name=MODEL_NAME, linear_weights_key=LINEAR_WEIGHTS_KEY
    )


@pytest.mark.skip_global_cleanup
def test_1_to_1(llm, hf_model):
    text_pair = [TEXTS_1[0], TEXTS_2[0]]

    hf_outputs = hf_model.predict([text_pair]).tolist()
    vllm_outputs = [
        output.outputs.score for output in llm.score(text_pair[0], text_pair[1])
    ]

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)


@pytest.mark.skip_global_cleanup
def test_1_to_n(llm, hf_model):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[0], TEXTS_2[1]],
    ]

    hf_outputs = hf_model.predict(text_pairs).tolist()
    vllm_outputs = [output.outputs.score for output in llm.score(TEXTS_1[0], TEXTS_2)]

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.mark.skip_global_cleanup
def test_n_to_n(llm, hf_model):
    text_pairs = [
        [TEXTS_1[0], TEXTS_2[0]],
        [TEXTS_1[1], TEXTS_2[1]],
    ]

    hf_outputs = hf_model.predict(text_pairs).tolist()
    vllm_outputs = [output.outputs.score for output in llm.score(TEXTS_1, TEXTS_2)]

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


def test_token_embed(llm):
    outputs = llm.encode(PROMPT, pooling_task="token_embed", use_tqdm=False)
    assert len(outputs) == 1
    assert outputs[0].outputs.data.shape == (9, COLBERT_DIM)
