# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# flake8: noqa
"""Tests Model Optimizer nvfp4 models."""

import pytest

from vllm.platforms import current_platform

EAGER = [True, False]

SM_100_NVFP4_BACKENDS = [
    "flashinfer_cutedsl",
    "flashinfer_cudnn",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "torch",
]


@pytest.mark.parametrize("model", ["nvidia/Llama-3.1-8B-Instruct-NVFP4"])
@pytest.mark.parametrize("eager", EAGER)
@pytest.mark.parametrize(
    "backend",
    [
        "emulation",
        "flashinfer_cutedsl",
        "flashinfer_cudnn",
        "flashinfer_trtllm",  # the small seq_len ensures trtllm_8x4_layout backend is used
        "flashinfer_cutlass",
        "torch",
    ],
)
def test_nvfp4(vllm_runner, model, eager, backend):
    if backend == "flashinfer_cutedsl" and not (
        current_platform.is_device_capability_family(100)
    ):
        pytest.skip("The flashinfer_cutedsl backend is only supported on SM10x")

    if (
        not current_platform.has_device_capability(100)
        and backend in SM_100_NVFP4_BACKENDS
    ):
        pytest.skip(
            f"The backend {backend} is not supported with current_platform.has_device_capability(100) == False"
        )

    with vllm_runner(model, enforce_eager=eager, linear_backend=backend) as llm:
        output = llm.generate_greedy(["1 2 3 4 5"], max_tokens=2)
    assert output[0][1] == "1 2 3 4 5 6"


@pytest.mark.parametrize(
    "model",
    [
        "nvidia/Qwen3-30B-A3B-NVFP4",
        "RedHatAI/Qwen3-30B-A3B-NVFP4",
    ],
)
@pytest.mark.parametrize("backend", ["emulation"])
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="NVFP4 MOE emulation is only useful on AMD Instinct MI3xx",
)
def test_nvfp4_moe(vllm_runner, model, backend):
    with vllm_runner(
        model,
        moe_backend=backend,
        linear_backend=backend,
        load_format="dummy",
        hf_overrides={"num_hidden_layers": 2},
    ) as llm:
        _ = llm.generate_greedy(["1 2 3 4 5"], max_tokens=2)
