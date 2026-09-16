# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from ._model_correctness import ATTN_BACKEND, run_models


@pytest.mark.parametrize("model", ["hmellor/tiny-random-Gemma2ForCausalLM"])
@pytest.mark.parametrize("backend", ATTN_BACKEND)
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize("async_scheduling", [True, False])
@pytest.mark.parametrize("model_executor", ["uni", "mp"])
@pytest.mark.parametrize("enable_prompt_embeds", [True, False])
def test_models(
    hf_runner,
    model: str,
    backend: str,
    max_tokens: int,
    enforce_eager: bool,
    async_scheduling: bool,
    model_executor: str,
    enable_prompt_embeds: bool,
) -> None:
    run_models(
        hf_runner,
        model,
        backend,
        max_tokens,
        enforce_eager,
        async_scheduling,
        model_executor,
        enable_prompt_embeds,
    )
