# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level parity: ReplaySSM standard decode vs the baseline SSM kernel."""

import pytest

from ...models.utils import check_logprobs_close
from ...utils import large_gpu_mark

# Mamba2 (Nemotron) and GDN (Qwen3.5) hybrids.
MODELS = [
    pytest.param(
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", marks=large_gpu_mark(min_gb=40)
    ),
    pytest.param("Qwen/Qwen3.5-4B", marks=large_gpu_mark(min_gb=40)),
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
]


@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_decode_matches_baseline(vllm_runner, model_name):
    # ReplaySSM reconstructs the state in different fp arithmetic, so greedy ids
    # can diverge at a near-tie; compare logprobs, not exact ids.
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
    )
    with vllm_runner(model_name, **common) as llm:
        baseline = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    with vllm_runner(
        model_name, use_replayssm=True, replayssm_buffer_len=16, **common
    ) as llm:
        replay = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline",
        name_1="replayssm",
    )
