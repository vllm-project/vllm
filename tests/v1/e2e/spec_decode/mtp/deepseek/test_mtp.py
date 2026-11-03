# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.platforms import current_platform

from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        pytest.param(
            ("mtp", "ZixiQi/DeepSeek-V3-4layers-MTP-FP8", 1, None),
            False,
            0.0,
            marks=pytest.mark.skipif(
                current_platform.is_device_capability_family(100),
                reason="DeepSeek MTP: TRTLLM MoE top_k check fails on Blackwell",
            ),
        ),  # dummy model
    ],
    ids=["deepseek"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )
