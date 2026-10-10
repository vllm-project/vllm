# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Granite 4 hybrid correctness, split out of hybrid/test_hybrid.py::test_models.

Its output is sensitive to hardware-specific Triton SSD autotuning
(https://github.com/vllm-project/vllm/issues/25194), so it runs on its own
compatibility job instead of the Hybrid job.
"""

import pytest

from tests.models.language.generation.hybrid._hybrid_models import check_models

pytestmark = pytest.mark.hybrid_model


@pytest.mark.parametrize("model", ["ibm-granite/granite-4.0-tiny-preview"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    gpu_memory_cleared,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    check_models(
        hf_runner, vllm_runner, example_prompts, model, max_tokens, num_logprobs
    )
