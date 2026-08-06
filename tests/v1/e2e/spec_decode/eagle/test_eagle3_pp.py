# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE3 speculative decoding under pipeline parallelism."""

import pytest
import torch

from tests.utils import multi_gpu_marks, multi_gpu_test
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

PROMPTS = [
    "The capital of France is",
    "2 + 2 equals",
    "In one word, the color of the sky is",
]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model,draft",
    [
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            "nm-testing/Llama3_2_1B_speculator.eagle3",
        ),
    ],
)
def test_eagle3_pipeline_parallel_greedy_parity(
    model: str,
    draft: str,
):
    """Greedy outputs with EAGLE3+PP=2 must match the same topology without spec."""
    pp_size = 2
    common = dict(
        model=model,
        tensor_parallel_size=1,
        pipeline_parallel_size=pp_size,
        max_model_len=512,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)

    ref = LLM(**common)
    ref_outs = ref.generate(PROMPTS, sampling)
    del ref
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    spec = LLM(
        **common,
        speculative_config={
            "method": "eagle3",
            "model": draft,
            "num_speculative_tokens": 3,
        },
    )
    spec_outs = spec.generate(PROMPTS, sampling)
    del spec
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    for ref_out, spec_out in zip(ref_outs, spec_outs):
        assert ref_out.outputs[0].text == spec_out.outputs[0].text, (
            f"PP={pp_size} greedy mismatch:\n"
            f"  ref={ref_out.outputs[0].text!r}\n"
            f"  spec={spec_out.outputs[0].text!r}"
        )
