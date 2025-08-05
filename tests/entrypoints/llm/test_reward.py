# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch

from vllm import LLM, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory

from ...models.utils import softmax

MODEL_NAME = "internlm/internlm2-1_8b-reward"

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
              trust_remote_code=True,
              seed=0)

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)

        del llm

    cleanup_dist_env_and_memory()


@pytest.mark.skip_global_cleanup
def test_pooling_params(llm: LLM):

    def get_outputs(softmax):
        outputs = llm.reward(prompts,
                             pooling_params=PoolingParams(softmax=softmax),
                             use_tqdm=False)
        return torch.cat([x.outputs.data for x in outputs])

    default = get_outputs(softmax=None)
    w_softmax = get_outputs(softmax=True)
    wo_softmax = get_outputs(softmax=False)

    assert torch.allclose(default, w_softmax,
                          atol=1e-2), "Default should use softmax."
    assert not torch.allclose(w_softmax, wo_softmax,
                              atol=1e-2), "wo_softmax should not use softmax."
    assert torch.allclose(
        softmax(wo_softmax), w_softmax,
        atol=1e-2), "w_softmax should be close to softmax(wo_softmax)."
