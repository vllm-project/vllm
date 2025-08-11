# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch

from vllm import LLM, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory

from ...models.utils import softmax

MODEL_NAME = "jason9693/Qwen2.5-1.5B-apeach"

prompts = ["The chef prepared a delicious meal."]


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(model=MODEL_NAME,
              max_num_batched_tokens=32768,
              tensor_parallel_size=1,
              gpu_memory_utilization=0.75,
              enforce_eager=True,
              seed=0)

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)

        del llm

    cleanup_dist_env_and_memory()


@pytest.mark.skip_global_cleanup
def test_pooling_params(llm: LLM):

    def get_outputs(activation):
        outputs = llm.classify(
            prompts,
            pooling_params=PoolingParams(activation=activation),
            use_tqdm=False)
        return torch.tensor([x.outputs.probs for x in outputs])

    default = get_outputs(activation=None)
    w_activation = get_outputs(activation=True)
    wo_activation = get_outputs(activation=False)

    assert torch.allclose(default, w_activation,
                          atol=1e-2), "Default should use activation."
    assert not torch.allclose(
        w_activation, wo_activation,
        atol=1e-2), "wo_activation should not use activation."
    assert torch.allclose(
        softmax(wo_activation), w_activation, atol=1e-2
    ), "w_activation should be close to activation(wo_activation)."


def test_encode_api(llm: LLM):
    err_msg = "pooling_task must be one of.+"
    with pytest.raises(ValueError, match=err_msg):
        llm.encode(prompts, use_tqdm=False)
