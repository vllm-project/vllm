# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._hybrid_models import _test_models

pytestmark = pytest.mark.hybrid_model

MODELS = ["ibm-granite/granite-4.0-tiny-preview"]


@pytest.mark.parametrize("model", MODELS)
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
