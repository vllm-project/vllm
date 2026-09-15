# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from .._voxtral import run_models_with_multiple_audios


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models_with_multiple_audios(
    vllm_runner,
    audio_assets,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    run_models_with_multiple_audios(
        vllm_runner, audio_assets, dtype, max_tokens, num_logprobs
    )
