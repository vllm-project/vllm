# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import pytest
import torch

from tests.models.utils import softmax
from vllm import LLM, ClassificationRequestOutput, PoolingParams
from vllm.tasks import PoolingTask

MODEL_NAME = "jason9693/Qwen2.5-1.5B-apeach"

prompt = "The chef prepared a delicious meal."
prompt_token_ids = [785, 29706, 10030, 264, 17923, 15145, 13]
num_labels = 2


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        max_model_len=None,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        enforce_eager=True,
        seed=0,
        enable_chunked_prefill=None,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.mark.skip_global_cleanup
def test_str_prompts(llm: LLM):
    outputs = llm.classify(prompt, use_tqdm=False)
    assert len(outputs) == 1
    assert isinstance(outputs[0], ClassificationRequestOutput)
    assert outputs[0].prompt_token_ids == prompt_token_ids
    assert len(outputs[0].outputs.probs) == num_labels


@pytest.mark.skip_global_cleanup
def test_token_ids_prompts(llm: LLM):
    outputs = llm.classify([prompt_token_ids], use_tqdm=False)
    assert len(outputs) == 1
    assert isinstance(outputs[0], ClassificationRequestOutput)
    assert outputs[0].prompt_token_ids == prompt_token_ids
    assert len(outputs[0].outputs.probs) == num_labels


@pytest.mark.skip_global_cleanup
def test_list_prompts(llm: LLM):
    outputs = llm.classify([prompt, prompt_token_ids], use_tqdm=False)
    assert len(outputs) == 2
    for i in range(len(outputs)):
        assert isinstance(outputs[i], ClassificationRequestOutput)
        assert outputs[i].prompt_token_ids == prompt_token_ids
        assert len(outputs[i].outputs.probs) == num_labels


@pytest.mark.skip_global_cleanup
def test_pooling_params(llm: LLM):
    def get_outputs(use_activation):
        outputs = llm.classify(
            prompt,
            pooling_params=PoolingParams(use_activation=use_activation),
            use_tqdm=False,
        )
        return torch.tensor([x.outputs.probs for x in outputs])

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


@pytest.mark.skip_global_cleanup
def test_score_api(llm: LLM):
    err_msg = "Scoring API is only enabled for num_labels == 1."
    with pytest.raises(ValueError, match=err_msg):
        llm.score("ping", "pong", use_tqdm=False)


@pytest.mark.parametrize("task", ["embed", "token_embed", "token_classify", "plugin"])
def test_unsupported_tasks(llm: LLM, task: PoolingTask):
    if task == "plugin":
        err_msg = "No IOProcessor plugin installed."
    elif task == "token_classify":
        err_msg = "Try switching the model's pooling_task via.+"
    else:
        err_msg = "Embedding API is not supported by this model.+"
    with pytest.raises(ValueError, match=err_msg):
        llm.encode(prompt, pooling_task=task, use_tqdm=False)
