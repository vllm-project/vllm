# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams

from .utils import compute_acceptance_rate


@pytest.mark.distributed(num_gpus=2)
def test_qwen3_5_dflash_pipeline_parallel(monkeypatch, vllm_runner):
    """PP must preserve generation and relay auxiliary states used by DFlash."""
    assert torch.accelerator.device_count() >= 2, "This test requires two GPUs"
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = [
        "The capital of France is",
        "The sequence of integers from one to ten is: 1, 2, 3,",
        "Repeat the word hello: hello hello hello hello",
        "Q: What is two plus two?\nA:",
    ]
    sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, logprobs=5)
    baseline = None
    for pp_size, use_dflash in ((1, False), (2, False), (1, True), (2, True)):
        spec_config = (
            {
                "method": "dflash",
                "model": "z-lab/Qwen3.5-4B-DFlash",
                "num_speculative_tokens": 15,
            }
            if use_dflash
            else None
        )
        name = f"PP={pp_size}, DFlash={use_dflash}"
        with vllm_runner(
            "Qwen/Qwen3.5-4B",
            dtype="bfloat16",
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=1,
            distributed_executor_backend="mp",
            speculative_config=spec_config,
            language_model_only=True,
            enforce_eager=True,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            max_model_len=512,
            max_num_seqs=4,
            gpu_memory_utilization=0.8,
            disable_log_stats=False,
            seed=0,
        ) as runner:
            assert runner.llm.llm_engine.vllm_config.use_v2_model_runner
            outputs = runner.generate_w_logprobs(prompts, sampling)
            assert len(outputs) == len(prompts), name
            for tokens, text, logprobs in outputs:
                assert len(tokens) == len(logprobs) == 32, name
                assert text, name
            if use_dflash:
                acceptance = compute_acceptance_rate(runner.llm.get_metrics())
                print(f"{name}: acceptance_rate={acceptance:.4f}")
                assert 0 < acceptance <= 1, name
        if baseline is None:
            baseline = outputs
        else:
            check_logprobs_close(
                outputs_0_lst=baseline,
                outputs_1_lst=outputs,
                name_0="PP=1, DFlash=False",
                name_1=name,
            )
