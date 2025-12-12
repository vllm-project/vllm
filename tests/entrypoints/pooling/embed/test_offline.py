# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch
import torch.nn.functional as F

from vllm import LLM, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        "Encoder self-attention is not implemented on ROCm.", allow_module_level=True
    )

MODEL_NAME = "intfloat/multilingual-e5-small"

prompts = ["The chef prepared a delicious meal."]


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
def test_token_embed(llm: LLM):
    outputs = llm.encode(prompts, pooling_task="token_embed", use_tqdm=False)
    multi_vector = outputs[0].outputs.data
    assert multi_vector.shape == (11, 384)


def test_pooling_params(llm: LLM):
    def get_outputs(normalize):
        outputs = llm.embed(
            prompts, pooling_params=PoolingParams(normalize=normalize), use_tqdm=False
        )
        return torch.tensor([x.outputs.embedding for x in outputs])

    default = get_outputs(normalize=None)
    w_normal = get_outputs(normalize=True)
    wo_normal = get_outputs(normalize=False)

    assert torch.allclose(default, w_normal, atol=1e-2), "Default should use normal."
    assert not torch.allclose(w_normal, wo_normal, atol=1e-2), (
        "wo_normal should not use normal."
    )
    assert torch.allclose(w_normal, F.normalize(wo_normal, p=2, dim=-1), atol=1e-2), (
        "w_normal should be close to normal(wo_normal)."
    )
