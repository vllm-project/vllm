# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level parity: ReplaySSM standard decode vs the baseline SSM kernel."""

import pytest

from ...models.utils import check_logprobs_close
from ...utils import large_gpu_mark, multi_gpu_test

# Mamba2 (Nemotron-3) hybrid.
MAMBA2_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
MODELS = [
    pytest.param(MAMBA2_MODEL, marks=large_gpu_mark(min_gb=40)),
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
]


def _check_replayssm_parity(vllm_runner, model_name, *, tensor_parallel_size=1):
    # Compare logprobs, not greedy ids: ReplaySSM's fp arithmetic can flip a
    # near-tie. Baseline and ReplaySSM run at the same TP, so TP numerics are
    # common-mode and only ReplaySSM varies.
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        tensor_parallel_size=tensor_parallel_size,
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


@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_decode_matches_baseline(vllm_runner, model_name):
    _check_replayssm_parity(vllm_runner, model_name)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model_name", [MAMBA2_MODEL])
def test_replayssm_decode_matches_baseline_tp2(vllm_runner, model_name):
    # Tensor-parallel correctness: ReplaySSM's caches and checkpoint state are
    # sharded per rank, so TP2 decode must still match the baseline at TP2.
    _check_replayssm_parity(vllm_runner, model_name, tensor_parallel_size=2)
