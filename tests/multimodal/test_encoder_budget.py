# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import DeviceConfig, SchedulerConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget

from ..models.utils import build_model_context

pytestmark = pytest.mark.cpu_test

MODEL = "ibm-granite/granite-speech-3.3-2b"


def _budget(
    skip_mm_profiling: bool, audio_seconds: int | None = None
) -> MultiModalBudget:
    audio_limit = (
        1 if audio_seconds is None else {"count": 1, "length": audio_seconds * 16000}
    )
    ctx = build_model_context(
        MODEL,
        model_config_kwargs={
            "max_model_len": 32768,
            "skip_mm_profiling": skip_mm_profiling,
        },
        limit_mm_per_prompt={"audio": audio_limit},
    )
    vllm_config = VllmConfig(
        model_config=ctx.model_config,
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=8192,
            max_num_seqs=256,
            max_model_len=32768,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig(device="cpu"),
    )
    return MultiModalBudget(vllm_config, MULTIMODAL_REGISTRY, enable_cache=False)


def test_profile_fills_the_budget_with_hinted_dummies():
    budget = _budget(skip_mm_profiling=False, audio_seconds=30)
    dummy = budget.processor.get_dummy_mm_inputs({"audio": 1})
    dummy_tokens = dummy["mm_placeholders"]["audio"][0].get_num_embeds()
    profiled = len(budget.get_dummy_encoder_profile_inputs())

    assert budget.encoder_profile_tokens == profiled * dummy_tokens
    assert (
        budget.encoder_compute_budget - dummy_tokens
        < budget.encoder_profile_tokens
        <= budget.encoder_compute_budget
    )


def test_encoder_profile_tokens_without_profiling():
    budget = _budget(skip_mm_profiling=True)

    assert budget.encoder_profile_tokens == budget.encoder_compute_budget
