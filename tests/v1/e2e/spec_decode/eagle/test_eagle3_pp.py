# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.utils import multi_gpu_test
from tests.v1.e2e.spec_decode.utils import compute_acceptance_len
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DRAFT = "nm-testing/Llama3_2_1B_speculator.eagle3"
# Enough tokens that one greedy path changing under compilation can't move the mean
TOPICS = ["the ocean", "a robot", "Rome", "volcanoes"]
TEMPLATES = [
    "Write a short story about {}.",
    "Explain {} to a child.",
    "List five facts about {}:",
    "Q: What is the history of {}?\nA:",
]
PROMPTS = [template.format(topic) for topic in TOPICS for template in TEMPLATES]

ACCEPTANCE_TOLERANCE = 0.95


def _run(pp_size: int, model_impl: str) -> float:
    llm = LLM(
        model=MODEL,
        model_impl=model_impl,
        tensor_parallel_size=1,
        pipeline_parallel_size=pp_size,
        max_model_len=512,
        gpu_memory_utilization=0.45,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
        speculative_config={
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": 3,
        },
    )
    try:
        llm.generate(
            PROMPTS,
            SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
        )
        acceptance = compute_acceptance_len(llm.get_metrics())
        assert acceptance > 1
        return acceptance
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("model_impl", ["vllm", "transformers"])
def test_eagle3_pipeline_parallel_acceptance(model_impl: str):
    baseline = _run(1, model_impl)
    for pp_size in (2, 4):
        parallel = _run(pp_size, model_impl)
        assert parallel >= baseline * ACCEPTANCE_TOLERANCE, (
            f"PP={pp_size} acceptance regressed: {parallel:.3f} < "
            f"{baseline:.3f} * {ACCEPTANCE_TOLERANCE}"
        )
