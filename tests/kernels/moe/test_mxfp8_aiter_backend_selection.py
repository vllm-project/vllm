# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE backend selection for the AITER FlyDSL kernel (gfx950).

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
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (  # noqa: E402
    _AITER_SWIGLU_ALPHA,
    _AITER_SWIGLU_BETA,
    _CLAMPED_SILU_ALPHA,
    _CLAMPED_SILU_BETA,
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
    # Default to the SwiGLU-OAI variant so is_supported_config doesn't reject the
    # config on activation grounds; see _hy4_config for the clamped-SiLU variant.
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


def test_gfx942_picks_emulation():
    """Flydsl unusable (e.g. gfx942, no FlyDSL support) -> native Triton
    dot_scaled backend wins instead."""
    with patch(f"{_AITER_MOD}.current_platform.supports_mx", return_value=False):
        backend, experts_cls = select_mxfp8_moe_backend(_config())
    assert backend is Fp8MoeBackend.EMULATION
    assert experts_cls is Mxfp8EmulationTritonExperts


def _hy4_config(swiglu_limit: float | None = 10.0):
    """A Hy4-style config: plain SiLU + a clamp limit, alpha/beta left unset.

    ``HYV4MoEFused`` passes only ``swiglu_limit`` to ``FusedMoEFactory``, so
    alpha/beta stay None and mean the ``SiluAndMulWithClamp`` identity
    (alpha=1.0, beta=0.0) -- i.e. ``silu(clamp(gate)) * clamp(up)``.
    """
    cfg = make_dummy_moe_config(
        num_experts=256,
        experts_per_token=8,
        hidden_dim=4096,
        activation=MoEActivation.SILU,
    )
    return dataclasses.replace(
        cfg, swiglu_alpha=None, swiglu_beta=None, swiglu_limit=swiglu_limit
    )


def _supported(cfg):
    return AiterMxfp8Experts.is_supported_config(
        AiterMxfp8Experts,
        cfg,
        kMxfp8Static,
        kMxfp8Dynamic,
        FusedMoEActivationFormat.Standard,
    )


def test_hy4_clamped_silu_is_supported():
    """Hy4's clamped SwiGLU (silu(clamp(g))*clamp(u)) must be accepted."""
    with _gfx950(), _aiter_moe_enabled(True):
        supported, reason = _supported(_hy4_config())
    assert supported, reason


def test_swigluoai_still_supported():
    """The pre-existing SwiGLU-OAI path must keep working."""
    with _gfx950(), _aiter_moe_enabled(True):
        supported, reason = _supported(_config())
    assert supported, reason


def test_hy4_clamped_silu_requires_limit():
    """Clamped SiLU without a limit is ambiguous -> reject rather than silently
    running an unclamped activation."""
    with _gfx950(), _aiter_moe_enabled(True):
        supported, reason = _supported(_hy4_config(swiglu_limit=None))
    assert not supported
    assert "swiglu_limit" in reason


def test_swigluoai_without_alpha_beta_rejected():
    """A SwiGLU-OAI config that never set alpha/beta must be rejected, not
    silently run as clamped SiLU (which is a different activation)."""
    cfg = dataclasses.replace(
        _config(), swiglu_alpha=None, swiglu_beta=None, swiglu_limit=7.0
    )
    with _gfx950(), _aiter_moe_enabled(True):
        supported, reason = _supported(cfg)
    assert not supported
    assert "swigluoai_uninterleave" in reason


def test_silu_with_oai_alpha_beta_rejected():
    """Conversely, plain SILU carrying OAI's alpha/beta is not clamped SiLU."""
    cfg = dataclasses.replace(
        _hy4_config(), swiglu_alpha=_AITER_SWIGLU_ALPHA, swiglu_beta=_AITER_SWIGLU_BETA
    )
    with _gfx950(), _aiter_moe_enabled(True):
        supported, reason = _supported(cfg)
    assert not supported
    assert "silu" in reason


def test_unsupported_alpha_beta_rejected():
    """An alpha/beta pair that is neither SwiGLU-OAI nor clamped SiLU is rejected."""
    cfg = dataclasses.replace(
        _hy4_config(), swiglu_alpha=1.234, swiglu_beta=_CLAMPED_SILU_BETA
    )
    with _gfx950(), _aiter_moe_enabled(True):
        supported, reason = _supported(cfg)
    assert not supported
    assert "1.234" in reason


def test_unsupported_activation_rejected():
    """An activation outside the supported set is rejected."""
    cfg = dataclasses.replace(_hy4_config(), activation=MoEActivation.GELU)
    with _gfx950(), _aiter_moe_enabled(True):
        supported, reason = _supported(cfg)
    assert not supported
    assert "gelu" in reason


def test_hy4_selects_aiter_backend():
    """End to end through the oracle: a Hy4 config auto-picks the FlyDSL backend."""
    with (
        patch(f"{_AITER_MOD}.current_platform.supports_mx", return_value=True),
        _aiter_moe_enabled(True),
    ):
        backend, experts_cls = select_mxfp8_moe_backend(_hy4_config())
    assert backend is Fp8MoeBackend.AITER_MXFP8
    assert experts_cls is AiterMxfp8Experts


def test_hy4_falls_back_to_triton_without_aiter():
    """``_CLAMPED_SILU_ALPHA`` is the documented identity; without aiter the
    Hy4 config must still resolve to a non-AITER backend."""
    assert _CLAMPED_SILU_ALPHA == 1.0
    with patch(f"{_AITER_MOD}.current_platform.supports_mx", return_value=False):
        backend, _ = select_mxfp8_moe_backend(_hy4_config())
    assert backend is not Fp8MoeBackend.AITER_MXFP8
