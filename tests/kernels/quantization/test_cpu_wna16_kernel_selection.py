# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm.model_executor.kernels.linear as linear_mod
from vllm.model_executor.kernels.linear import (
    CPUWNA16LinearKernel,
    MPLinearLayerConfig,
    ZentorchWNA16LinearKernel,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision import (
    cpu as cpu_mod,
)
from vllm.model_executor.kernels.linear.mixed_precision import (
    zentorch as zentorch_mod,
)
from vllm.platforms import CpuArchEnum, PlatformEnum
from vllm.scalar_type import scalar_types

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _config(
    *, act_type: torch.dtype = torch.float16, has_g_idx: bool = False
) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(128, 128),
        partition_weight_shape=(128, 128),
        weight_type=scalar_types.uint4,
        act_type=act_type,
        group_size=32,
        zero_points=True,
        has_g_idx=has_g_idx,
    )


def _set_cpu_platform(monkeypatch, architecture: CpuArchEnum, *, zen=False) -> None:
    platform = SimpleNamespace(
        _enum=PlatformEnum.CPU,
        is_cpu=lambda: True,
        is_zen_cpu=lambda: zen,
        get_cpu_architecture=lambda: architecture,
    )
    monkeypatch.setattr(cpu_mod, "current_platform", platform)
    monkeypatch.setattr(zentorch_mod, "current_platform", platform)
    monkeypatch.setattr(linear_mod, "current_platform", platform)


def _select_cpu_wna16(monkeypatch, config: MPLinearLayerConfig):
    monkeypatch.setitem(
        linear_mod._POSSIBLE_KERNELS,
        PlatformEnum.CPU,
        [CPUWNA16LinearKernel],
    )
    return choose_mp_linear_kernel(config, compute_capability=0)


@pytest.mark.parametrize("operator_present", [False, True])
def test_w4a16_requires_registered_operator(monkeypatch, operator_present):
    _set_cpu_platform(monkeypatch, CpuArchEnum.X86)
    monkeypatch.setattr(cpu_mod.envs, "VLLM_CPU_INT4_W4A8", False)
    namespace = SimpleNamespace()
    if operator_present:
        namespace.cpu_gemm_wna16 = object()
    with patch.object(cpu_mod.torch.ops, "_C", namespace):
        if operator_present:
            assert _select_cpu_wna16(monkeypatch, _config()) is CPUWNA16LinearKernel
            return
        with pytest.raises(ValueError, match="cpu_gemm_wna16 is not registered"):
            _select_cpu_wna16(monkeypatch, _config())


@pytest.mark.parametrize("has_g_idx", [False, True])
def test_riscv_w4a8_only_bypasses_wna16_without_g_idx(monkeypatch, has_g_idx):
    _set_cpu_platform(monkeypatch, CpuArchEnum.RISCV)
    monkeypatch.setattr(cpu_mod.envs, "VLLM_CPU_INT4_W4A8", True)
    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: False)
    with patch.object(cpu_mod.torch.ops, "_C", SimpleNamespace()):
        config = _config(act_type=torch.bfloat16, has_g_idx=has_g_idx)
        if not has_g_idx:
            assert _select_cpu_wna16(monkeypatch, config) is CPUWNA16LinearKernel
            return
        with pytest.raises(ValueError, match="cpu_gemm_wna16 is not registered"):
            _select_cpu_wna16(monkeypatch, config)


def test_zentorch_uses_its_own_operator_gate(monkeypatch):
    _set_cpu_platform(monkeypatch, CpuArchEnum.X86, zen=True)
    monkeypatch.setattr(cpu_mod.envs, "VLLM_CPU_INT4_W4A8", False)
    monkeypatch.setattr(zentorch_mod, "has_zentorch_op", lambda _: True)
    with patch.object(cpu_mod.torch.ops, "_C", SimpleNamespace()):
        assert ZentorchWNA16LinearKernel.can_implement(_config()) == (True, None)
