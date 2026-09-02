# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from tests.entrypoints.pooling.scoring.util import EncoderScoringHfRunner

MODEL_NAME = "intfloat/multilingual-e5-small"
PROMPT = "The chef prepared a delicious meal."
EMBEDDING_SIZE = 384

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
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        max_model_len=None,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
        enable_chunked_prefill=None,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.fixture(scope="module")
def hf_model():
    return EncoderScoringHfRunner(MODEL_NAME)


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


def test_embed(llm):
    outputs = llm.encode(PROMPT, pooling_task="embed", use_tqdm=False)
    assert len(outputs) == 1
    assert len(outputs[0].outputs.data) == EMBEDDING_SIZE
