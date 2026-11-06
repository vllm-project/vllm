# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import (
    get_attn_backend_list_based_on_platform,
    single_gpu_only,
)
from vllm import SamplingParams
from vllm.platforms import current_platform

from .utils import _run_eagle_correctness


@single_gpu_only
@pytest.mark.skipif(
    current_platform.is_device_capability_family(100),
    reason="DeepSeek head_dim=192 not supported on SM100/SM110 (Blackwell)",
)
@pytest.mark.parametrize(
    [
        "model_setup",
        "mm_enabled",
        "enable_chunked_prefill",
        "model_impl",
        "expected_accuracy_threshold",
    ],
    [
        (
            (
                "eagle",
                "eagle618/deepseek-v3-random",
                "eagle618/eagle-deepseek-v3-random",
                1,
            ),
            False,
            False,
            "auto",
            0.0,
        ),
    ],
    ids=["deepseek_eagle"],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness_light(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
    vllm_runner,
):
    _run_eagle_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        enable_chunked_prefill,
        model_impl,
        attn_backend,
        vllm_runner,
    )


@single_gpu_only
@pytest.mark.parametrize(
    [
        "model_setup",
        "mm_enabled",
        "enable_chunked_prefill",
        "model_impl",
        "expected_accuracy_threshold",
    ],
    [
        (
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "auto",
            0.8,
        ),
        pytest.param(
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "transformers",
            0.8,
            # TODO(hmellor): figure out why memory usage is so high
            marks=pytest.mark.skipif(
                not current_platform.is_rocm(),
                reason="Feature is experimental and uses too much memory in CUDA CI",
            ),
        ),
        (
            (
                "eagle3",
                "Qwen/Qwen3-VL-8B-Instruct",
                "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3",
                1,
            ),
            False,
            False,
            "auto",
            0.8,
        ),
        pytest.param(
            (
                "eagle3",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Rayzl/qwen2.5-vl-7b-eagle3-sgl",
                1,
            ),
            False,
            False,
            "auto",
            0.7,
            # TODO: Re-measure the reference threshold when re-enabling this case.
            # Text-only mode starts but loses target correctness with ROCm/TRITON;
            # full multimodal profiling currently fails during engine startup.
            marks=pytest.mark.skip(
                reason="Qwen2.5-VL Eagle3 is not yet reliable in this test setup"
            ),
        ),
        (
            (
                "eagle3",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
            False,
            "auto",
            0.7,
        ),
    ],
    ids=[
        "qwen3_eagle3",
        "qwen3_eagle3-transformers",
        "qwen3_vl_eagle3",
        "qwen2_5_vl_eagle3",
        "llama3_eagle3",
    ],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness_medium(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
    vllm_runner,
):
    _run_eagle_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        enable_chunked_prefill,
        model_impl,
        attn_backend,
        vllm_runner,
    )
