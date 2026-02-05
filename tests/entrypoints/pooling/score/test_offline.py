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


def test_pooling_params(llm: LLM):
    def get_outputs(use_activation):
        queries = "What is the capital of France?"
        documents = "The capital of France is Paris."

        outputs = llm.score(
            queries,
            documents,
            pooling_params=PoolingParams(use_activation=use_activation),
            use_tqdm=False,
        )
        return torch.tensor([x.outputs.score for x in outputs])

    default = get_outputs(use_activation=None)
    w_activation = get_outputs(use_activation=True)
    wo_activation = get_outputs(use_activation=False)

    assert torch.allclose(
        default, w_activation, atol=1e-2
    ), "Default should use activation."
    assert not torch.allclose(
        w_activation, wo_activation, atol=1e-2
    ), "wo_activation should not use activation."
    assert torch.allclose(
        softmax(wo_activation), w_activation, atol=1e-2
    ), "w_activation should be close to activation(wo_activation)."
