# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch
import torch.nn.functional as F

from vllm import LLM, EmbeddingRequestOutput, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform
from vllm.tasks import PoolingTask

MODEL_NAME = "intfloat/multilingual-e5-small"

prompt = "The chef prepared a delicious meal."
prompt_token_ids = [0, 581, 21861, 133888, 10, 8, 150, 60744, 109911, 5, 2]
embedding_size = 384


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
    assert embedding_size == llm.model_config.embedding_size

    yield weakref.proxy(llm)

    del llm
    cleanup_dist_env_and_memory()


@pytest.mark.skip_global_cleanup
def test_str_prompts(llm: LLM):
    outputs = llm.embed(prompt, use_tqdm=False)
    assert len(outputs) == 1
    assert isinstance(outputs[0], EmbeddingRequestOutput)
    assert outputs[0].prompt_token_ids == prompt_token_ids
    assert len(outputs[0].outputs.embedding) == embedding_size


@pytest.mark.skip_global_cleanup
def test_token_ids_prompts(llm: LLM):
    outputs = llm.embed([prompt_token_ids], use_tqdm=False)
    assert len(outputs) == 1
    assert isinstance(outputs[0], EmbeddingRequestOutput)
    assert outputs[0].prompt_token_ids == prompt_token_ids
    assert len(outputs[0].outputs.embedding) == embedding_size


@pytest.mark.skip_global_cleanup
def test_list_prompts(llm: LLM):
    outputs = llm.embed([prompt, prompt_token_ids], use_tqdm=False)
    assert len(outputs) == 2
    for i in range(len(outputs)):
        assert isinstance(outputs[i], EmbeddingRequestOutput)
        assert outputs[i].prompt_token_ids == prompt_token_ids
        assert len(outputs[i].outputs.embedding) == embedding_size


@pytest.mark.skip_global_cleanup
def test_pooling_params(llm: LLM):
    def get_outputs(normalize):
        outputs = llm.embed(
            [prompt],
            pooling_params=PoolingParams(use_activation=normalize),
            use_tqdm=False,
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


@pytest.mark.parametrize("task", ["token_embed", "classify", "token_classify"])
def test_unsupported_tasks(llm: LLM, task: PoolingTask):
    if task == "token_embed":
        err_msg = "Try switching the model's pooling_task via.+"
    else:
        err_msg = "Classification API is not supported by this model.+"

    with pytest.raises(ValueError, match=err_msg):
        llm.encode(prompt, pooling_task=task, use_tqdm=False)
