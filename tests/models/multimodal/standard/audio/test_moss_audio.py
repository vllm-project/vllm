# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from .._moss_audio import (
    CORE_MODEL,
    HF_ACCURACY_SKIP_REASON,
    PARALLEL_SMOKE_CASES,
    run_moss_audio_generation_smoke,
    run_moss_audio_hf_vllm_accuracy,
    run_moss_audio_parallel_smoke,
)


@pytest.mark.core_model
def test_moss_audio_generation_smoke(vllm_runner) -> None:
    run_moss_audio_generation_smoke(vllm_runner)


@pytest.mark.skip(reason=HF_ACCURACY_SKIP_REASON)
@pytest.mark.parametrize("model", [CORE_MODEL])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [8])
@pytest.mark.parametrize("num_logprobs", [5])
def test_moss_audio_hf_vllm_accuracy(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    run_moss_audio_hf_vllm_accuracy(
        hf_runner, vllm_runner, model, dtype, max_tokens, num_logprobs
    )


@pytest.mark.core_model
@pytest.mark.parametrize("parallel_kwargs", PARALLEL_SMOKE_CASES)
def test_moss_audio_parallel_smoke(vllm_runner, parallel_kwargs) -> None:
    run_moss_audio_parallel_smoke(vllm_runner, parallel_kwargs)
