# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE backend selection for the AITER kernels: FlyDSL (gfx950) and
moe_gemm_a8w8 (gfx942).

GPU-free: mocks the platform (gfx950) and the aiter fused-MoE enable flag, then
exercises the oracle so the FlyDSL backend is auto-picked when usable (including
under expert parallelism, since apply() forwards the expert_map as aiter's
expert_mask) and skipped (native fallback) when the device is unsupported or the
aiter runtime is disabled.
"""

import dataclasses
from unittest.mock import patch

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_a8w8_moe import (  # noqa: E402
    AiterMxfp8A8W8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (  # noqa: E402
    _AITER_SWIGLU_ALPHA,
    _AITER_SWIGLU_BETA,
    AiterMxfp8Experts,
)
from vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe import (  # noqa: E402
    Mxfp8EmulationTritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (  # noqa: E402
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (  # noqa: E402
    Fp8MoeBackend,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (  # noqa: E402
    _BACKEND_NAME_MAP,
    _SUPPORTED_BACKENDS,
    _mxfp8_backend_to_kernel_cls,
    _select_kernel_cls,
    select_mxfp8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kMxfp8Dynamic,
    kMxfp8Static,
)

from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402  # isort: skip

_AITER_MOD = "vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe"


def _config(ep_size: int = 1):
    # AiterMxfp8Experts hardcodes SwiGLU-OAI: match its required activation and
    # alpha/beta so is_supported_config doesn't reject the config on those grounds.
    cfg = make_dummy_moe_config(
        num_experts=128,
        experts_per_token=4,
        hidden_dim=6144,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )
    cfg = dataclasses.replace(
        cfg, swiglu_alpha=_AITER_SWIGLU_ALPHA, swiglu_beta=_AITER_SWIGLU_BETA
    )
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
        f"{_AITER_MOD}.current_platform",
        is_rocm=lambda: True,
        supports_mx=lambda: True,
    )


def _aiter_moe_enabled(present: bool):
    return patch.object(rocm_aiter_ops, "is_fused_moe_enabled", return_value=present)


def test_aiter_mxfp8_registered():
    """The FlyDSL backend is auto-selectable and reachable via --moe-backend aiter."""
    assert Fp8MoeBackend.AITER_MXFP8 in _SUPPORTED_BACKENDS
    assert _BACKEND_NAME_MAP["aiter"] is Fp8MoeBackend.AITER_MXFP8
    assert _mxfp8_backend_to_kernel_cls(Fp8MoeBackend.AITER_MXFP8) == [
        AiterMxfp8Experts
    ]


@pytest.mark.parametrize("ep_size", [1, 2])
def test_ep_supported(ep_size):
    """FlyDSL accepts both TP and EP: apply() forwards expert_map as expert_mask."""
    assert (
        AiterMxfp8Experts._supports_parallel_config(
            _config(ep_size).moe_parallel_config
        )
        is True
    )


@pytest.mark.parametrize(
    "present,ep_size,supported,reason_substr",
    [
        (True, 1, True, None),  # gfx950 + aiter MoE + TP -> selectable
        (True, 2, True, None),  # gfx950 + aiter MoE + EP -> selectable (expert_mask)
        (
            False,
            1,
            False,
            "does not support current device",
        ),  # disabled -> not selected
    ],
)
def test_is_supported_config(present, ep_size, supported, reason_substr):
    with _gfx950(), _aiter_moe_enabled(present):
        ok, reason = AiterMxfp8Experts.is_supported_config(
            AiterMxfp8Experts,
            _config(ep_size),
            kMxfp8Static,
            kMxfp8Dynamic,
            FusedMoEActivationFormat.Standard,
        )
    assert ok is supported
    if reason_substr is not None:
        assert reason_substr in reason


def test_explicit_moe_backend_aiter():
    """--moe-backend aiter: returns FlyDSL when usable (TP or EP), else a clear
    ValueError when the aiter MoE runtime is disabled."""
    with _gfx950(), _aiter_moe_enabled(True):
        assert (
            _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(1))
            is AiterMxfp8Experts
        )
        assert (
            _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(2))
            is AiterMxfp8Experts
        )
    with (
        _gfx950(),
        _aiter_moe_enabled(False),
        pytest.raises(ValueError, match="does not support current device"),
    ):
        _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(1))


def test_gfx950_picks_aiter():
    """Auto-select on real ROCm hardware with aiter MoE enabled -> FlyDSL wins."""
    with (
        patch(f"{_AITER_MOD}.current_platform.supports_mx", return_value=True),
        _aiter_moe_enabled(True),
    ):
        backend, experts_cls = select_mxfp8_moe_backend(_config())
    assert backend is Fp8MoeBackend.AITER_MXFP8
    assert experts_cls is AiterMxfp8Experts


def test_no_aiter_picks_emulation():
    """Neither AITER kernel usable (no MX hardware, not gfx942) -> EMULATION."""
    with (
        patch(f"{_AITER_MOD}.current_platform.supports_mx", return_value=False),
        _gfx942(False),
    ):
        backend, experts_cls = select_mxfp8_moe_backend(_config())
    assert backend is Fp8MoeBackend.EMULATION
    assert experts_cls is Mxfp8EmulationTritonExperts


# --- gfx942: AITER moe_gemm_a8w8 ---------------------------------------------

_A8W8_ROCM = "vllm.platforms.rocm"


def _gfx942(on: bool):
    return patch(f"{_A8W8_ROCM}.on_gfx942", return_value=on)


def _a8w8_config(**overrides):
    cfg = make_dummy_moe_config(
        num_experts=128,
        experts_per_token=8,
        hidden_dim=6144,
        activation=MoEActivation.SILU,
    )
    fse = overrides.pop("is_fused_shared_expert_requested", False)
    overrides.setdefault("routing_method", RoutingMethodType.DeepSeekV3)
    cfg = dataclasses.replace(cfg, **overrides)
    # __post_init__ derives this from the env, so set it afterwards.
    cfg.is_fused_shared_expert_requested = fse
    return cfg


def _a8w8_supported(cfg):
    return AiterMxfp8A8W8ExpertsMonolithic.is_supported_config(
        AiterMxfp8A8W8ExpertsMonolithic,
        cfg,
        kMxfp8Static,
        kMxfp8Dynamic,
        FusedMoEActivationFormat.Standard,
    )


def test_a8w8_registered():
    assert Fp8MoeBackend.AITER_MXFP8_A8W8 in _SUPPORTED_BACKENDS
    order = list(_SUPPORTED_BACKENDS)
    assert order.index(Fp8MoeBackend.AITER_MXFP8_A8W8) < order.index(
        Fp8MoeBackend.EMULATION
    )
    assert _BACKEND_NAME_MAP["aiter_mxfp8_a8w8"] is Fp8MoeBackend.AITER_MXFP8_A8W8
    assert _mxfp8_backend_to_kernel_cls(Fp8MoeBackend.AITER_MXFP8_A8W8) == [
        AiterMxfp8A8W8ExpertsMonolithic
    ]


def test_gfx942_picks_a8w8():
    with (
        patch(f"{_AITER_MOD}.current_platform.supports_mx", return_value=False),
        _gfx942(True),
        _aiter_moe_enabled(True),
    ):
        backend, experts_cls = select_mxfp8_moe_backend(_a8w8_config())
    assert backend is Fp8MoeBackend.AITER_MXFP8_A8W8
    assert experts_cls is AiterMxfp8A8W8ExpertsMonolithic


@pytest.mark.parametrize(
    "gfx942,aiter_moe,overrides,reason",
    [
        (True, True, {}, None),
        (False, True, {}, "current device"),  # e.g. gfx950
        (True, False, {}, "current device"),  # VLLM_ROCM_USE_AITER_MOE=0
        (True, True, {"has_bias": True}, "bias"),
        (True, True, {"is_fused_shared_expert_requested": True}, "shared experts"),
        (True, True, {"routing_method": RoutingMethodType.Llama4}, "routing method"),
    ],
)
def test_a8w8_is_supported_config(gfx942, aiter_moe, overrides, reason):
    with _gfx942(gfx942), _aiter_moe_enabled(aiter_moe):
        ok, why = _a8w8_supported(_a8w8_config(**overrides))
    assert ok is (reason is None)
    if reason is not None:
        assert reason in why


def test_a8w8_rejects_ep():
    cfg = _a8w8_config()
    cfg = dataclasses.replace(
        cfg,
        moe_parallel_config=dataclasses.replace(
            cfg.moe_parallel_config, ep_size=2, use_ep=True
        ),
    )
    with _gfx942(True), _aiter_moe_enabled(True):
        ok, why = _a8w8_supported(cfg)
    assert not ok and "parallel config" in why


def test_explicit_moe_backend_a8w8_raises_when_unsupported():
    with (
        _gfx942(False),
        _aiter_moe_enabled(True),
        pytest.raises(ValueError, match="does not support current device"),
    ):
        _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8_A8W8, _a8w8_config())
