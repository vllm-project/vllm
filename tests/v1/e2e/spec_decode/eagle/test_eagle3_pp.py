# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.utils import multi_gpu_test
from tests.v1.e2e.spec_decode.utils import (
    Messages,
    compute_acceptance_len,
    get_test_prompts,
)
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DRAFT = "nm-testing/Llama3_2_1B_speculator.eagle3"
PROMPTS = [
    "The capital of France is",
    "2 + 2 equals",
    "In one word, the color of the sky is",
    "Q: If a train travels 60 miles in 1.5 hours, what is its average speed?\nA:",
]

ACCEPTANCE_TOLERANCE = 0.95


def _run(
    pp_size: int,
    model: str = MODEL,
    draft: str = DRAFT,
    max_model_len: int = 512,
    max_tokens: int = 32,
    chat_prompts: list[Messages] | None = None,
) -> float:
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        pipeline_parallel_size=pp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.45,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
        speculative_config={
            "method": "eagle3",
            "model": draft,
            "num_speculative_tokens": 3,
        },
    )
    try:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=True
        )
        if chat_prompts is None:
            llm.generate(PROMPTS, sampling_params)
        else:
            llm.chat(chat_prompts, sampling_params)
        acceptance = compute_acceptance_len(llm.get_metrics())
        assert acceptance > 1
        return acceptance
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=4)
def test_eagle3_pipeline_parallel_acceptance():
    baseline = _run(1)
    for pp_size in (2, 4):
        parallel = _run(pp_size)
        assert parallel >= baseline * ACCEPTANCE_TOLERANCE, (
            f"PP={pp_size} acceptance regressed: {parallel:.3f} < "
            f"{baseline:.3f} * {ACCEPTANCE_TOLERANCE}"
        )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model,draft",
    [
        # The text model inherits PP support from Qwen2Model but has its own
        # forward, which must send this stage's aux hidden states.
        ("Qwen/Qwen3-VL-4B-Instruct", "AngelSlim/Qwen3-VL-4B-Instruct_eagle3"),
        ("openai/gpt-oss-20b", "RedHatAI/gpt-oss-20b-speculator.eagle3"),
    ],
)
def test_eagle3_pipeline_parallel_targets(model, draft):
    """Targets whose own forward passes aux hidden states between PP stages."""
    # Greedy outputs at PP=1 and PP=2 can diverge after a few tokens, so the
    # acceptance lengths are compared over many prompts.
    prompts = get_test_prompts(mm_enabled=False, num_prompts=64)
    baseline = _run(1, model, draft, 2048, 128, prompts)
    parallel = _run(2, model, draft, 2048, 128, prompts)
    assert parallel >= baseline * ACCEPTANCE_TOLERANCE, (
        f"PP=2 acceptance regressed: {parallel:.3f} < "
        f"{baseline:.3f} * {ACCEPTANCE_TOLERANCE}"
    )
