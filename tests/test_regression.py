"""Containing tests that check for regressions in vLLM's behavior.

It should include tests that are reported by users and making sure they
will never happen again.

"""
import gc

import torch

from vllm import LLM, SamplingParams


def test_duplicated_ignored_sequence_group():
    """https://github.com/vllm-project/vllm/issues/1655"""

    sampling_params = SamplingParams(temperature=0.01,
                                     top_p=0.1,
                                     max_tokens=256)
    llm = LLM(model="facebook/opt-125m",
              max_num_batched_tokens=4096,
              tensor_parallel_size=1)
    prompts = ["This is a short prompt", "This is a very long prompt " * 1000]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(prompts) == len(outputs)


def test_max_tokens_none():
    sampling_params = SamplingParams(temperature=0.01,
                                     top_p=0.1,
                                     max_tokens=None)
    llm = LLM(model="facebook/opt-125m",
              max_num_batched_tokens=4096,
              tensor_parallel_size=1)
    prompts = ["Just say hello!"]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(prompts) == len(outputs)


def test_gc():
    llm = LLM("facebook/opt-125m", enforce_eager=True)
    del llm

    gc.collect()
    torch.cuda.empty_cache()

    # The memory allocated for model and KV cache should be released.
    # The memory allocated for PyTorch and others should be less than 50MB.
    # Usually, it's around 10MB.
    allocated = torch.cuda.memory_allocated()
    assert allocated < 50 * 1024 * 1024


def test_model_from_modelscope(monkeypatch):
    # model: https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary
    MODELSCOPE_MODEL_NAME = "qwen/Qwen1.5-0.5B-Chat"
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "True")
    try:
        llm = LLM(model=MODELSCOPE_MODEL_NAME)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        outputs = llm.generate(prompts, sampling_params)
        assert len(outputs) == 4
    finally:
        monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
