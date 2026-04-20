# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    BaseOAITritonExperts,
    OAITritonMxfp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe import utils as fused_moe_utils
from vllm.platforms.interface import DeviceCapability


pytestmark = pytest.mark.skip_global_cleanup


class FakePlatform:

    def __init__(self, *, is_rocm: bool, capability: tuple[int, int]):
        self._is_rocm = is_rocm
        self._capability = capability

    def is_cuda_alike(self) -> bool:
        return True

    def is_rocm(self) -> bool:
        return self._is_rocm

    def is_cuda(self) -> bool:
        return not self._is_rocm

    def get_device_capability(self):
        return DeviceCapability(*self._capability)


def test_supports_triton_mxfp4_device_respects_platform_specific_ceiling():
    assert fused_moe_utils.supports_triton_mxfp4_device(
        FakePlatform(is_rocm=False, capability=(10, 0))
    )
    assert not fused_moe_utils.supports_triton_mxfp4_device(
        FakePlatform(is_rocm=False, capability=(11, 0))
    )
    assert fused_moe_utils.supports_triton_mxfp4_device(
        FakePlatform(is_rocm=True, capability=(11, 5))
    )
    assert not fused_moe_utils.supports_triton_mxfp4_device(
        FakePlatform(is_rocm=True, capability=(12, 0))
    )


def test_oai_triton_experts_allow_rocm_rdna35(monkeypatch):
    monkeypatch.setattr(
        fused_moe_utils,
        "current_platform",
        FakePlatform(is_rocm=True, capability=(11, 5)),
    )

    assert BaseOAITritonExperts._supports_current_device()
    assert OAITritonMxfp4ExpertsMonolithic._supports_current_device()


def test_oai_triton_experts_keep_cuda_sm110_disabled(monkeypatch):
    monkeypatch.setattr(
        fused_moe_utils,
        "current_platform",
        FakePlatform(is_rocm=False, capability=(11, 0)),
    )

    assert not BaseOAITritonExperts._supports_current_device()
    assert not OAITritonMxfp4ExpertsMonolithic._supports_current_device()
