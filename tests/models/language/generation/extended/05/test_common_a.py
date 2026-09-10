# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._common_models import _test_models
from tests.utils import large_gpu_mark
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("swiss-ai/Apertus-8B-Instruct-2509"),  # apertus
        pytest.param(
            "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",  # hyperclovax
            marks=[large_gpu_mark(min_gb=32)],
        ),
        pytest.param(
            "zai-org/chatglm3-6b",  # chatglm (text-only)
        ),
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
