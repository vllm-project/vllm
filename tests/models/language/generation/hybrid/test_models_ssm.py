# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._hybrid_models import SSM_MODELS, _test_models

pytestmark = pytest.mark.hybrid_model


@pytest.mark.parametrize("model", SSM_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    _test_models(
        hf_runner,
        vllm_runner,
        example_prompts,
        monkeypatch,
        model,
        max_tokens,
        num_logprobs,
    )
