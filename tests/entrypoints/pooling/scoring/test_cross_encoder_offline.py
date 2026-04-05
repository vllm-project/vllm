# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch

from tests.models.utils import softmax
from vllm import LLM, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

MODEL_NAME = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
PROMPT = "The chef prepared a delicious meal."
TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]


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
def hf_model(hf_runner):
    return hf_runner(MODEL_NAME, is_cross_encoder=True)


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

    vllm_outputs = [output.outputs.score for output in llm.score(TEXTS_1[0], TEXTS_2)]
    hf_outputs = hf_model.predict(text_pairs).tolist()

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

    vllm_outputs = [output.outputs.score for output in llm.score(TEXTS_1, TEXTS_2)]
    hf_outputs = hf_model.predict(text_pairs).tolist()

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.mark.skip_global_cleanup
def test_classify(llm):
    outputs = llm.encode(PROMPT, pooling_task="classify", use_tqdm=False)
    assert len(outputs) == 1
    assert len(outputs[0].outputs.data) == 1


@pytest.mark.skip_global_cleanup
def test_max_tokens_per_doc_offline(llm: LLM):
    """Test max_tokens_per_doc via PoolingParams.extra_kwargs (offline)."""
    long_doc = "The capital of France is Paris. " * 20

    # Without truncation
    outputs_no_limit = llm.score(
        TEXTS_1[0],
        long_doc,
        use_tqdm=False,
    )

    # With truncation via extra_kwargs
    outputs_with_limit = llm.score(
        TEXTS_1[0],
        long_doc,
        pooling_params=PoolingParams(
            extra_kwargs={"max_tokens_per_doc": 10}
        ),
        use_tqdm=False,
    )

    assert len(outputs_no_limit) == 1
    assert len(outputs_with_limit) == 1

    # Truncated version should have fewer prompt tokens
    no_limit_tokens = len(outputs_no_limit[0].prompt_token_ids)
    with_limit_tokens = len(outputs_with_limit[0].prompt_token_ids)
    assert with_limit_tokens < no_limit_tokens


def test_pooling_params(llm: LLM):
    def get_outputs(use_activation):
        outputs = llm.score(
            TEXTS_1[0],
            TEXTS_2[0],
            pooling_params=PoolingParams(use_activation=use_activation),
            use_tqdm=False,
        )
        return torch.tensor([x.outputs.score for x in outputs])

    default = get_outputs(use_activation=None)
    w_activation = get_outputs(use_activation=True)
    wo_activation = get_outputs(use_activation=False)

    assert torch.allclose(default, w_activation, atol=1e-2), (
        "Default should use activation."
    )
    assert not torch.allclose(w_activation, wo_activation, atol=1e-2), (
        "wo_activation should not use activation."
    )
    assert torch.allclose(softmax(wo_activation), w_activation, atol=1e-2), (
        "w_activation should be close to activation(wo_activation)."
    )
