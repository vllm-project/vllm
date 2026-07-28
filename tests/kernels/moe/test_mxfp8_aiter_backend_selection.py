# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE backend selection for the AITER FlyDSL kernel (gfx950).

GPU-free: mocks the platform (gfx950) and the ``flydsl`` package check, then
exercises the oracle so the FlyDSL backend is auto-picked when usable (including
under expert parallelism, since apply() forwards the expert_map as aiter's
expert_mask) and skipped (native fallback) when the device/package is missing.
"""

import dataclasses
import sys
import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (  # noqa: E402
    AiterMxfp8Experts,
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
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kMxfp8Dynamic,
    kMxfp8Static,
)

_AITER_MOD = "vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe"


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
        f"{_AITER_MOD}.current_platform",
        is_rocm=lambda: True,
        supports_mx=lambda: True,
    )


def _flydsl_installed(present: bool):
    return patch(f"{_AITER_MOD}.is_aiter_mxfp8_moe_available", return_value=present)


def test_aiter_mxfp8_registered():
    """The FlyDSL backend is auto-selectable and reachable via --moe-backend aiter."""
    assert Fp8MoeBackend.AITER_MXFP8 in _SUPPORTED_BACKENDS
    assert _BACKEND_NAME_MAP["aiter"] is Fp8MoeBackend.AITER_MXFP8
    assert _mxfp8_backend_to_kernel_cls(Fp8MoeBackend.AITER_MXFP8) == [
        AiterMxfp8Experts
    ]


def test_triton_selectable():
    assert _BACKEND_NAME_MAP["triton"] is Fp8MoeBackend.TRITON_MXFP8
    # Not auto-selected (only reachable explicitly), so FlyDSL still wins auto.
    assert Fp8MoeBackend.TRITON_MXFP8 not in _SUPPORTED_BACKENDS


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
        (True, 1, True, None),  # gfx950 + flydsl + TP -> selectable
        (True, 2, True, None),  # gfx950 + flydsl + EP -> selectable (expert_mask)
        (False, 1, False, "flydsl package"),  # package missing -> clear reason
    ],
)
def test_is_supported_config(present, ep_size, supported, reason_substr):
    with _gfx950(), _flydsl_installed(present):
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
    ValueError when the flydsl package is missing."""
    with _gfx950(), _flydsl_installed(True):
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
        _flydsl_installed(False),
        pytest.raises(ValueError, match="flydsl package"),
    ):
        _select_kernel_cls(Fp8MoeBackend.AITER_MXFP8, _config(1))


@pytest.mark.parametrize("output_kind", ["owning", "view", "noncontiguous"])
def test_aiter_mxfp8_rebinds_output_only_when_safe(monkeypatch, output_kind):
    """Owning outputs adopt AITER storage; views retain their storage."""
    from vllm._aiter_ops import rocm_aiter_ops

    def _enum_value(value):
        return types.SimpleNamespace(value=value)

    fake_aiter = types.ModuleType("aiter")
    fake_aiter.__dict__["ActivationType"] = types.SimpleNamespace(Swiglu=_enum_value(1))
    fake_aiter.__dict__["QuantType"] = types.SimpleNamespace(per_1x32=_enum_value(2))
    fake_aiter_ops = types.ModuleType("aiter.ops")
    fake_flydsl = types.ModuleType("aiter.ops.flydsl")
    fake_moe_common = types.ModuleType("aiter.ops.flydsl.moe_common")
    fake_moe_common.__dict__["GateMode"] = types.SimpleNamespace(
        INTERLEAVE=_enum_value("interleave")
    )
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_aiter_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake_flydsl)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.moe_common", fake_moe_common)

    experts = object.__new__(AiterMxfp8Experts)
    experts.moe_config = types.SimpleNamespace(rocm_aiter_fmoe_enabled=False)
    experts.quant_config = types.SimpleNamespace(gemm1_clamp_limit=None)
    experts.w1_scale_val = None
    experts.w2_scale_val = None

    result = torch.randn(4, 16, dtype=torch.bfloat16)
    if output_kind == "owning":
        output = torch.empty_like(result)
    elif output_kind == "view":
        output = torch.empty(8, 16, dtype=result.dtype)[:4]
    else:
        output = torch.empty(4, 32, dtype=result.dtype)[:, ::2]
    original_output_ptr = output.data_ptr()
    original_output_stride = output.stride()

    def _fake_fused_moe(*args, **kwargs):
        assert kwargs["output_dtype"] == output.dtype
        return result

    monkeypatch.setattr(rocm_aiter_ops, "fused_moe", _fake_fused_moe)

    experts.apply(
        output=output,
        hidden_states=torch.empty_like(result),
        w1=torch.empty(1),
        w2=torch.empty(1),
        topk_weights=torch.ones(4, 2),
        topk_ids=torch.zeros(4, 2, dtype=torch.int32),
        activation=None,
        global_num_experts=1,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    if output_kind == "owning":
        assert output.data_ptr() == result.data_ptr()
        assert output.data_ptr() != original_output_ptr
    else:
        assert output.data_ptr() == original_output_ptr
        assert output.data_ptr() != result.data_ptr()
        assert output.stride() == original_output_stride
    assert torch.equal(output, result)
