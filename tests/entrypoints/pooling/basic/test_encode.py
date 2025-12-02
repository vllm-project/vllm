# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from vllm import LLM, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        "Encoder self-attention is not implemented on ROCm.", allow_module_level=True
    )

MODEL_NAME = "intfloat/multilingual-e5-small"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

TOKEN_IDS = [
    # Using ID={0, 1, 2, 3} results in NaN values,
    # so we add this offset of 1000
    [1000],
    [1000, 1001],
    [1000, 1002, 1001],
    [1000, 1003, 1001, 1002],
]


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.mark.skip_global_cleanup
def test_multiple_pooling_params(llm: LLM):
    pooling_params = [
        PoolingParams(),
        PoolingParams(),
        PoolingParams(),
        PoolingParams(),
    ]

    # Multiple PoolingParams should be matched with each prompt
    outputs = llm.encode(PROMPTS, pooling_params=pooling_params, pooling_task="embed")
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.encode(
            PROMPTS, pooling_params=pooling_params[:3], pooling_task="embed"
        )

    # Single PoolingParams should be applied to every prompt
    single_pooling_params = PoolingParams()
    outputs = llm.encode(
        PROMPTS, pooling_params=single_pooling_params, pooling_task="embed"
    )
    assert len(PROMPTS) == len(outputs)

    # pooling_params is None, default params should be applied
    outputs = llm.encode(PROMPTS, pooling_params=None, pooling_task="embed")
    assert len(PROMPTS) == len(outputs)


def test_right_side_truncation(llm: LLM):
    # Embeddings models should truncate the end of the prompt
    tokenizer = llm.get_tokenizer()
    assert tokenizer.truncation_side == "right"
