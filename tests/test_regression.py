# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Containing tests that check for regressions in vLLM's behavior.

It should include tests that are reported by users and making sure they
will never happen again.

"""

import gc

import pytest
import torch

from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "distilbert/distilgpt2",
            marks=[
                *([large_gpu_mark(min_gb=80)] if current_platform.is_rocm() else []),
            ],
        ),
    ],
)
def test_max_tokens_none(model):
    sampling_params = SamplingParams(temperature=0.01, top_p=0.1, max_tokens=None)
    llm = LLM(
        model=model,
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
    )
    prompts = ["Just say hello!"]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(prompts) == len(outputs)


def test_gc():
    llm = LLM(model="distilbert/distilgpt2", enforce_eager=True)
    del llm

    gc.collect()
    torch.accelerator.empty_cache()

    # The memory allocated for model and KV cache should be released.
    # The memory allocated for PyTorch and others should be less than 50MB.
    # Usually, it's around 10MB.
    allocated = torch.accelerator.memory_allocated()
    assert allocated < 50 * 1024 * 1024


def test_model_from_modelscope(monkeypatch: pytest.MonkeyPatch):
    # model: https://www.modelscope.ai/models/qwen/Qwen1.5-0.5B-Chat
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        m.setenv("MODELSCOPE_DOMAIN", "www.modelscope.ai")
        # Don't use HF_TOKEN for ModelScope repos, otherwise it will fail
        # with 400 Client Error: Bad Request.
        m.setenv("HF_TOKEN", "")
        attn_backend = "TRITON_ATTN" if current_platform.is_rocm() else "auto"
        llm = LLM(model="qwen/Qwen1.5-0.5B-Chat", attention_backend=attn_backend)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        outputs = llm.generate(prompts, sampling_params)
        assert len(outputs) == 4
