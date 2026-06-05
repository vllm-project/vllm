# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Python unit tests for the MoRI EP dispatch/combine dtype plumbing.

These cover ``vllm...prepare_finalize.mori_dtypes`` which drives the MoRI EP
runtime configuration without touching any GPU / RDMA path.  The module is
intentionally free of ``import mori``, so these tests run on any host (no ROCm,
no ``mori`` binding).

Covered behaviour:

* ``combine_quant_type`` maps the combine enum to the exact string MoRI's
  ``EpDispatchCombineConfig.quant_type`` expects (drift here silently disables
  fp8 combine -> accuracy regression).
* ``resolve_mori_dtypes`` auto-selects (dispatch, combine) from the *weight*
  quant dtype (SGLang #21040 behaviour: mxfp4 weights -> fp4/fp8, fp8 weights
  -> fp8/bf16, else bf16/bf16).
* ``VLLM_ROCM_MORI_DISPATCH_DTYPE`` / ``VLLM_ROCM_MORI_COMBINE_DTYPE`` env
  overrides win over auto-detection, are case-insensitive, and fall back to the
  auto-detected value on garbage input.
* The enum string values are hard-pinned (they surface in logs/env values).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm.model_executor.layers.fused_moe.prepare_finalize.mori_dtypes import (
    CombineDtype,
    DispatchDtype,
    combine_quant_type,
    resolve_mori_dtypes,
)


@dataclass
class _FakeQuantConfig:
    """Minimal stand-in for ``FusedMoEQuantConfig``.

    ``resolve_mori_dtypes`` only reads these four booleans, so a lightweight
    fake avoids constructing a real (GPU/quant-method-bound) config.
    """

    use_mxfp4_w4a16: bool = False
    use_mxfp4_w4a4: bool = False
    use_mxfp4_w4a8: bool = False
    use_fp8_w8a8: bool = False


_MXFP4 = _FakeQuantConfig(use_mxfp4_w4a16=True)
_FP8 = _FakeQuantConfig(use_fp8_w8a8=True)
_UNQUANT = _FakeQuantConfig()

_FP4_PAIR = (DispatchDtype.fp4, CombineDtype.fp8)
_FP8_PAIR = (DispatchDtype.fp8, CombineDtype.bf16)
_BF16_PAIR = (DispatchDtype.bf16, CombineDtype.bf16)


# -- enum string values ------------------------------------------------------


def test_dispatch_dtype_string_values_stable():
    assert DispatchDtype.bf16.value == "bf16"
    assert DispatchDtype.fp8.value == "fp8"
    assert DispatchDtype.fp4.value == "fp4"


def test_combine_dtype_string_values_stable():
    assert CombineDtype.bf16.value == "bf16"
    assert CombineDtype.fp8.value == "fp8"
    assert CombineDtype.fp8_direct_cast.value == "fp8_direct_cast"


# -- combine enum -> mori string mapping -------------------------------------


@pytest.mark.parametrize(
    "combine_dtype, expected_str",
    [
        (CombineDtype.bf16, "none"),
        (CombineDtype.fp8, "fp8_blockwise"),
        (CombineDtype.fp8_direct_cast, "fp8_direct_cast"),
    ],
)
def test_combine_quant_type(combine_dtype, expected_str):
    assert combine_quant_type(combine_dtype) == expected_str


# -- auto-detect table (SGLang #21040) ---------------------------------------


@pytest.mark.parametrize(
    "quant_config, expected",
    [
        (None, _BF16_PAIR),
        (_UNQUANT, _BF16_PAIR),
        (_MXFP4, _FP4_PAIR),
        (_FakeQuantConfig(use_mxfp4_w4a4=True), _FP4_PAIR),
        (_FakeQuantConfig(use_mxfp4_w4a8=True), _FP4_PAIR),
        (_FP8, _FP8_PAIR),
    ],
)
def test_auto_detect(quant_config, expected, monkeypatch):
    # Ensure no env override leaks in from the host environment.
    monkeypatch.setenv("VLLM_ROCM_MORI_DISPATCH_DTYPE", "auto")
    monkeypatch.setenv("VLLM_ROCM_MORI_COMBINE_DTYPE", "auto")
    assert resolve_mori_dtypes(quant_config) == expected


# -- dispatch override -------------------------------------------------------


@pytest.mark.parametrize(
    "override, expected_dispatch",
    [
        ("auto", DispatchDtype.fp4),  # auto-detected from MXFP4 weights
        ("AUTO", DispatchDtype.fp4),  # case-insensitive (envs lowercases)
        ("bf16", DispatchDtype.bf16),
        ("fp8", DispatchDtype.fp8),
        ("fp4", DispatchDtype.fp4),
    ],
)
def test_dispatch_override_valid(override, expected_dispatch, monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_MORI_DISPATCH_DTYPE", override)
    monkeypatch.setenv("VLLM_ROCM_MORI_COMBINE_DTYPE", "auto")
    dispatch, _ = resolve_mori_dtypes(_MXFP4)
    assert dispatch == expected_dispatch


def test_dispatch_override_invalid_falls_back_to_auto(monkeypatch):
    """A nonsense override must not raise -- log + keep the auto value."""
    monkeypatch.setenv("VLLM_ROCM_MORI_DISPATCH_DTYPE", "garbage")
    monkeypatch.setenv("VLLM_ROCM_MORI_COMBINE_DTYPE", "auto")
    dispatch, _ = resolve_mori_dtypes(_MXFP4)
    assert dispatch == DispatchDtype.fp4


# -- combine override --------------------------------------------------------


@pytest.mark.parametrize(
    "override, expected_combine",
    [
        ("auto", CombineDtype.bf16),
        ("bf16", CombineDtype.bf16),
        ("fp8", CombineDtype.fp8),
        ("fp8_direct_cast", CombineDtype.fp8_direct_cast),
    ],
)
def test_combine_override_valid(override, expected_combine, monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_MORI_DISPATCH_DTYPE", "auto")
    monkeypatch.setenv("VLLM_ROCM_MORI_COMBINE_DTYPE", override)
    _, combine = resolve_mori_dtypes(_FP8)
    assert combine == expected_combine


def test_combine_override_invalid_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_MORI_DISPATCH_DTYPE", "auto")
    monkeypatch.setenv("VLLM_ROCM_MORI_COMBINE_DTYPE", "totally_made_up")
    _, combine = resolve_mori_dtypes(_FP8)
    assert combine == CombineDtype.bf16


def test_overrides_are_independent(monkeypatch):
    """Dispatch + combine overrides do not influence one another."""
    monkeypatch.setenv("VLLM_ROCM_MORI_DISPATCH_DTYPE", "fp4")
    monkeypatch.setenv("VLLM_ROCM_MORI_COMBINE_DTYPE", "fp8")
    dispatch, combine = resolve_mori_dtypes(None)  # bf16/bf16 by auto-detect
    assert dispatch == DispatchDtype.fp4
    assert combine == CombineDtype.fp8
