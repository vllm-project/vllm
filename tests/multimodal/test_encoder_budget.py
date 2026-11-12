# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import DeviceConfig, SchedulerConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget

from ..models.utils import build_model_context

pytestmark = pytest.mark.cpu_test

MODEL = "ibm-granite/granite-speech-3.3-2b"


def _budget(skip_mm_profiling: bool) -> MultiModalBudget:
    ctx = build_model_context(
        MODEL,
        model_config_kwargs={
            "max_model_len": 32768,
            "skip_mm_profiling": skip_mm_profiling,
        },
        limit_mm_per_prompt={"audio": 1},
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


def test_encoder_profile_tokens_match_the_profiled_batch():
    budget = _budget(skip_mm_profiling=False)
    profiled = len(budget.get_dummy_encoder_profile_inputs())

    assert (
        budget.encoder_profile_tokens == profiled * budget.mm_max_toks_per_item["audio"]
    )
    assert budget.encoder_profile_tokens < budget.encoder_compute_budget


def test_encoder_profile_tokens_without_profiling():
    budget = _budget(skip_mm_profiling=True)

    assert budget.encoder_profile_tokens == budget.encoder_compute_budget
