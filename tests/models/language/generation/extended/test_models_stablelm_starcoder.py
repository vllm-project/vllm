# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._common_models import _test_models
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("stabilityai/stablelm-3b-4e1t"),  # stablelm
        pytest.param("bigcode/starcoder2-3b"),  # starcoder2
        pytest.param("bigcode/tiny_starcoder_py"),  # gpt_bigcode
        pytest.param("EleutherAI/pythia-70m"),  # gpt_neox
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
@pytest.mark.parametrize("use_prompt_embeds", [True, False])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    use_rocm_aiter: bool,
    use_prompt_embeds: bool,
    monkeypatch,
) -> None:
    _test_models(
        hf_runner,
        vllm_runner,
        example_prompts,
        model,
        max_tokens,
        num_logprobs,
        use_rocm_aiter,
        use_prompt_embeds,
        monkeypatch,
    )
