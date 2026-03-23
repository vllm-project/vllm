# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from vllm import LLM, PoolingRequestOutput
from vllm.config import PoolerConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.tasks import PoolingTask

MODEL_NAME = "jason9693/Qwen2.5-1.5B-apeach"

prompt = "The chef prepared a delicious meal."
prompt_token_ids = [785, 29706, 10030, 264, 17923, 15145, 13]
num_labels = 2


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        pooler_config=PoolerConfig(pooling_task="token_classify"),
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
def test_str_prompts(llm: LLM):
    outputs = llm.encode(prompt, pooling_task="token_classify", use_tqdm=False)
    assert len(outputs) == 1
    assert isinstance(outputs[0], PoolingRequestOutput)
    assert outputs[0].prompt_token_ids == prompt_token_ids
    assert outputs[0].outputs.data.shape == (len(prompt_token_ids), num_labels)


@pytest.mark.skip_global_cleanup
def test_token_ids_prompts(llm: LLM):
    outputs = llm.encode(
        [prompt_token_ids], pooling_task="token_classify", use_tqdm=False
    )
    assert len(outputs) == 1
    assert isinstance(outputs[0], PoolingRequestOutput)
    assert outputs[0].prompt_token_ids == prompt_token_ids
    assert outputs[0].outputs.data.shape == (len(prompt_token_ids), num_labels)


@pytest.mark.skip_global_cleanup
def test_score_api(llm: LLM):
    err_msg = "Score API is only enabled for num_labels == 1."
    with pytest.raises(ValueError, match=err_msg):
        llm.score("ping", "pong", use_tqdm=False)


@pytest.mark.parametrize("task", ["classify", "embed", "token_embed", "plugin"])
def test_unsupported_tasks(llm: LLM, task: PoolingTask):
    if task == "classify":
        err_msg = "Try switching the model's pooling_task via.+"
    elif task in ["embed", "token_embed"]:
        err_msg = "Embedding API is not supported by this model.+"
    else:  # task == "plugin"
        err_msg = f"Unsupported task: '{task}' Supported tasks.+"

    with pytest.raises(ValueError, match=err_msg):
        llm.encode(prompt, pooling_task=task, use_tqdm=False)
