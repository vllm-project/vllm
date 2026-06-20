# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE backend selection for the AITER FlyDSL kernel (gfx950).

GPU-free: mocks the platform (gfx950) and the ``flydsl`` package check, then
exercises the oracle so the FlyDSL backend is auto-picked when usable, skipped
(native fallback) otherwise, and -- crucially -- NEVER selected under expert
parallelism (EP), which the kernel does not support.
"""

import dataclasses
from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.experts.flydsl_mxfp8_moe import (
    FlydslMxfp8Experts,
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

_FLYDSL_MOD = "vllm.model_executor.layers.fused_moe.experts.flydsl_mxfp8_moe"


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


def _gfx950():
    """Patch the platform so the device gate (gfx950 / MX) passes off-ROCm."""
    return patch.multiple(
        f"{_FLYDSL_MOD}.current_platform",
        is_rocm=lambda: True,
        supports_mx=lambda: True,
    )


def _flydsl_installed(present: bool):
    return patch(f"{_FLYDSL_MOD}.is_flydsl_mxfp8_moe_available", return_value=present)


def test_aiter_mxfp8_registered():
    """The FlyDSL backend is auto-selectable and reachable via --moe-backend aiter."""
    assert Fp8MoeBackend.AITER_MXFP8 in _SUPPORTED_BACKENDS
    assert _BACKEND_NAME_MAP["aiter"] is Fp8MoeBackend.AITER_MXFP8
    assert _mxfp8_backend_to_kernel_cls(Fp8MoeBackend.AITER_MXFP8) == [
        FlydslMxfp8Experts
    ]


@pytest.mark.parametrize("ep_size,expected", [(1, True), (2, False)])
def test_ep_guard(ep_size, expected):
    """FlyDSL must reject EP at selection time (apply() can't do EP)."""
    assert (
        FlydslMxfp8Experts._supports_parallel_config(
            _config(ep_size).moe_parallel_config
        )
        is expected
    )


@pytest.mark.parametrize(
    "present,ep_size,supported,reason_substr",
    [
        (True, 1, True, None),  # gfx950 + flydsl + no EP -> selectable
        (True, 2, False, "parallel config"),  # EP -> rejected (native fallback)
        (False, 1, False, "flydsl package"),  # package missing -> clear reason
    ],
)
def test_is_supported_config(present, ep_size, supported, reason_substr):
    with _gfx950(), _flydsl_installed(present):
        ok, reason = FlydslMxfp8Experts.is_supported_config(
            FlydslMxfp8Experts,
            _config(ep_size),
            kMxfp8Static,
            kMxfp8Dynamic,
            FusedMoEActivationFormat.Standard,
        )
    assert ok is supported
    if reason_substr is not None:
        assert reason_substr in reason


def test_explicit_moe_backend_aiter():
    """--moe-backend aiter: returns FlyDSL when usable, else a clear ValueError."""
    with _gfx950(), _flydsl_installed(True):
        assert (
            _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(1))
            is FlydslMxfp8Experts
        )
        with pytest.raises(ValueError, match="parallel config"):
            _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(2))
    with (
        _gfx950(),
        _flydsl_installed(False),
        pytest.raises(ValueError, match="flydsl package"),
    ):
        _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(1))
