# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ...standard._moss_audio import (
    EXTENDED_MODELS,
    HF_ACCURACY_SKIP_REASON,
    run_moss_audio_hf_vllm_accuracy,
)


@pytest.mark.skip(reason=HF_ACCURACY_SKIP_REASON)
@pytest.mark.parametrize("model", EXTENDED_MODELS)
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
