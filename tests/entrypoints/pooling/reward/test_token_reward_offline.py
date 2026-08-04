# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch

from tests.models.utils import softmax
from vllm import LLM, PoolingParams

MODEL_NAME = "internlm/internlm2-1_8b-reward"

prompts = ["The chef prepared a delicious meal."]


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        max_model_len=None,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        seed=0,
        enable_chunked_prefill=None,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.mark.skip_global_cleanup
def test_config(llm: LLM):
    vllm_config = llm.llm_engine.vllm_config
    assert vllm_config.cache_config.enable_prefix_caching
    assert vllm_config.scheduler_config.enable_chunked_prefill


def test_pooling_params(llm: LLM):
    def get_outputs(use_activation):
        outputs = llm.encode(
            prompts,
            pooling_params=PoolingParams(use_activation=use_activation),
            pooling_task="token_classify",
            use_tqdm=False,
        )
        return torch.cat([x.outputs.data for x in outputs])

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
