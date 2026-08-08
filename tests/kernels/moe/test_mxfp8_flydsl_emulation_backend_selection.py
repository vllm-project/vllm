# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-selection tests for the FlyDSL MXFP8-emulation MoE (gfx942).

GPU-free: mocks the platform and flydsl runtime to verify oracle routing.
"""

import contextlib
import dataclasses
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.experts.flydsl_emulation_moe import (
    FlydslEmulationExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    _BACKEND_NAME_MAP,
    _SUPPORTED_BACKENDS,
    _mxfp8_backend_to_kernel_cls,
    _select_kernel_cls,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kMxfp8Static,
)

_MOD = "vllm.model_executor.layers.fused_moe.experts.flydsl_emulation_moe"


def _config(ep_size: int = 1):
    cfg = make_dummy_moe_config(num_experts=128, experts_per_token=4, hidden_dim=6144)
    if ep_size != 1:
        cfg = dataclasses.replace(
            cfg,
            moe_parallel_config=dataclasses.replace(
                cfg.moe_parallel_config, ep_size=ep_size, use_ep=True
            ),
        )
    return cfg


_ORACLE_MOD = "vllm.model_executor.layers.fused_moe.oracle.mxfp8"


@contextlib.contextmanager
def _gfx942():
    """Mock ROCm gfx942 platform (no native MX, so the oracle dispatches
    AITER_MXFP8 to the BF16-emulation FlyDSL path)."""
    with (
        patch.multiple(f"{_MOD}.current_platform", is_rocm=lambda: True),
        patch(f"{_ORACLE_MOD}.current_platform.supports_mx", lambda: False),
        patch("vllm.platforms.rocm.on_gfx942", return_value=True),
    ):
        yield


def _flydsl_installed(present: bool):
    return patch(f"{_MOD}.is_flydsl_emulation_available", return_value=present)


def test_aiter_dispatches_to_flydsl_emulation_on_gfx942():
    """--moe-backend aiter maps to AITER_MXFP8, which on gfx942 (no native MX)
    resolves to the BF16-emulation FlyDSL experts. AITER_MXFP8 is auto-selectable
    (shared with the gfx950 native path added in #46184)."""
    assert _BACKEND_NAME_MAP["aiter"] is Fp8MoeBackend.AITER_MXFP8
    assert Fp8MoeBackend.AITER_MXFP8 in _SUPPORTED_BACKENDS
    with _gfx942():
        assert _mxfp8_backend_to_kernel_cls(Fp8MoeBackend.AITER_MXFP8) == [
            FlydslEmulationExperts
        ]


@pytest.mark.parametrize("ep_size,expected", [(1, True), (2, False)])
def test_ep_guard(ep_size, expected):
    """FlyDSL rejects EP."""
    assert (
        FlydslEmulationExperts._supports_parallel_config(
            _config(ep_size).moe_parallel_config
        )
        is expected
    )


@pytest.mark.parametrize(
    "present,ep_size,supported,reason_substr",
    [
        (True, 1, True, None),  # gfx942 + flydsl + no EP -> selectable
        (True, 2, False, "parallel config"),  # EP -> rejected (native fallback)
        (False, 1, False, "flydsl"),  # runtime missing -> clear reason
    ],
)
def test_is_supported_config(present, ep_size, supported, reason_substr):
    with _gfx942(), _flydsl_installed(present):
        ok, reason = FlydslEmulationExperts.is_supported_config(
            FlydslEmulationExperts,
            _config(ep_size),
            kMxfp8Static,
            kMxfp8Dynamic,
            FusedMoEActivationFormat.Standard,
        )
    assert ok is supported
    if reason_substr is not None:
        assert reason_substr in reason


def test_explicit_moe_backend_aiter():
    """--moe-backend aiter returns FlyDSL when usable, else ValueError."""
    with _gfx942(), _flydsl_installed(True):
        assert (
            _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(1))
            is FlydslEmulationExperts
        )
        with pytest.raises(ValueError, match="parallel config"):
            _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(2))
    with (
        _gfx942(),
        _flydsl_installed(False),
        pytest.raises(ValueError, match="flydsl"),
    ):
        _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(1))
