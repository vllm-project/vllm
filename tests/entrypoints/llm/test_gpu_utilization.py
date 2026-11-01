# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from ...utils import wait_for_memory_to_settle


def test_fixed_kv_cache_ignores_gpu_memory_utilization(monkeypatch):
    """A fixed cache must fit independently of the utilization fraction."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    outputs = []
    wait_for_memory_to_settle()
    for utilization in (1.0, 0.5):
        llm = LLM(
            "facebook/opt-125m",
            gpu_memory_utilization=utilization,
            kv_cache_memory_bytes=64 * 1024 * 1024,
            max_model_len=64,
            max_num_batched_tokens=64,
            max_num_seqs=1,
            enforce_eager=True,
            seed=0,
        )
        try:
            result = llm.generate(
                ["The capital of France is"],
                SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
            )
            outputs.append(result[0].outputs[0].token_ids)
        finally:
            llm.llm_engine.engine_core.shutdown()
            del llm
            cleanup_dist_env_and_memory()
            wait_for_memory_to_settle()

    assert len(outputs[0]) == 8
    assert outputs[1] == outputs[0]


@pytest.mark.parametrize("kv_cache_memory_bytes", [None, 0])
def test_auto_kv_cache_keeps_gpu_memory_utilization_guard(
    monkeypatch, kv_cache_memory_bytes
):
    """Unset and zero cache sizes retain automatic sizing's startup guard."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with pytest.raises(ValueError, match="less than desired GPU memory utilization"):
        LLM(
            "facebook/opt-125m",
            gpu_memory_utilization=1.0,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            max_model_len=64,
            enforce_eager=True,
        )


def test_gpu_memory_utilization(vllm_runner):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # makes sure gpu_memory_utilization is per-instance limit,
    # not a global limit
    with (
        vllm_runner(
            "facebook/opt-125m",
            gpu_memory_utilization=0.3,
            enforce_eager=True,
        ) as runner_0,
        vllm_runner(
            "facebook/opt-125m",
            gpu_memory_utilization=0.3,
            enforce_eager=True,
        ) as runner_1,
        vllm_runner(
            "facebook/opt-125m",
            gpu_memory_utilization=0.3,
            enforce_eager=True,
        ) as runner_2,
    ):
        for runner in (runner_0, runner_1, runner_2):
            outputs = runner.llm.generate(prompts, sampling_params)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
