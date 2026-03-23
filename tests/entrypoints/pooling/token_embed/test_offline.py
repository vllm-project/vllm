# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from vllm import LLM, PoolingRequestOutput
from vllm.config import PoolerConfig
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
        pooler_config=PoolerConfig(pooling_task="token_embed"),
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
    outputs = llm.encode(prompt, pooling_task="token_embed", use_tqdm=False)
    assert len(outputs) == 1
    assert isinstance(outputs[0], PoolingRequestOutput)
    assert outputs[0].outputs.data.shape == (11, 384)


@pytest.mark.skip_global_cleanup
def test_token_ids_prompts(llm: LLM):
    outputs = llm.encode([prompt_token_ids], pooling_task="token_embed", use_tqdm=False)
    assert len(outputs) == 1
    assert isinstance(outputs[0], PoolingRequestOutput)
    assert outputs[0].outputs.data.shape == (11, 384)


@pytest.mark.parametrize("task", ["embed", "classify", "token_classify", "plugin"])
def test_unsupported_tasks(llm: LLM, task: PoolingTask):
    if task == "embed":
        err_msg = "Try switching the model's pooling_task via.+"
    else:
        err_msg = "Classification API is not supported by this model.+"

    with pytest.raises(ValueError, match=err_msg):
        llm.encode(prompt, pooling_task=task, use_tqdm=False)
