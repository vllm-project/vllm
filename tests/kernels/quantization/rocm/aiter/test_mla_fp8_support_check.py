# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ROCm AITER MLA FP8 compatibility shim.

These tests do not try to validate MLA kernel numerics. They protect the small
runtime contract in ``vllm._aiter_ops`` that adapts vLLM to different AITER
versions:

- detect whether ``aiter.mla.mla_decode_fwd`` accepts ``q_scale`` and
  ``kv_scale``
- cache that probe result
- pass FP8 scale arguments only when the installed AITER supports them
- forward MLA work buffers consistently once decode enters the AITER path

Broader MLA decode and kernel-accuracy coverage lives in
``tests/kernels/rocm/aiter/test_rocm_aiter_mla.py``.
"""

from __future__ import annotations

import builtins
import inspect
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any, TypedDict

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-specific MLA AITER tests",
)


@pytest.fixture(autouse=True)
def reset_aiter_mla_fp8_support_cache():
    import vllm._aiter_ops as aiter_ops

    aiter_ops._AITER_MLA_SUPPORTS_FP8 = None
    yield
    aiter_ops._AITER_MLA_SUPPORTS_FP8 = None


class DecodeInputs(TypedDict):
    q: torch.Tensor
    kv_buffer: torch.Tensor
    o: torch.Tensor
    qo_indptr: torch.Tensor
    max_seqlen_qo: int
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_lens: torch.Tensor
    q_scale: torch.Tensor
    kv_scale: torch.Tensor


class WorkBufferInputs(TypedDict):
    work_meta_data: torch.Tensor
    work_indptr: torch.Tensor
    work_info_set: torch.Tensor
    reduce_indptr: torch.Tensor
    reduce_final_map: torch.Tensor
    reduce_partial_map: torch.Tensor


def _install_fake_aiter_mla(
    monkeypatch,
    mla_decode_fwd: Callable[..., object],
) -> None:
    aiter_pkg: Any = ModuleType("aiter")
    aiter_pkg.__path__ = []
    mla_mod: Any = ModuleType("aiter.mla")
    mla_mod.mla_decode_fwd = mla_decode_fwd
    aiter_pkg.mla = mla_mod
    monkeypatch.setitem(sys.modules, "aiter", aiter_pkg)
    monkeypatch.setitem(sys.modules, "aiter.mla", mla_mod)


def _make_decode_inputs() -> DecodeInputs:
    head_dim = 16
    return {
        "q": torch.zeros(2, head_dim),
        "kv_buffer": torch.zeros(3, head_dim),
        "o": torch.zeros(2, head_dim),
        "qo_indptr": torch.tensor([0, 1, 2], dtype=torch.int32),
        "max_seqlen_qo": 1,
        "kv_indptr": torch.tensor([0, 1, 3], dtype=torch.int32),
        "kv_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
        "kv_last_page_lens": torch.tensor([1, 1], dtype=torch.int32),
        "q_scale": torch.tensor(0.5),
        "kv_scale": torch.tensor(0.25),
    }


def _make_work_buffer_inputs() -> WorkBufferInputs:
    return {
        "work_meta_data": torch.zeros(4, dtype=torch.int32),
        "work_indptr": torch.tensor([0, 2, 4], dtype=torch.int32),
        "work_info_set": torch.zeros(2, dtype=torch.int32),
        "reduce_indptr": torch.tensor([0, 1, 2], dtype=torch.int32),
        "reduce_final_map": torch.zeros(2, dtype=torch.int32),
        "reduce_partial_map": torch.zeros(2, dtype=torch.int32),
    }


def test_aiter_mla_fp8_support_detects_scale_parameters(monkeypatch):
    """Newer AITER builds should be recognized as FP8-MLA-capable."""
    import vllm._aiter_ops as aiter_ops

    def mla_decode_fwd(*args, q_scale=None, kv_scale=None, **kwargs):
        return None

    _install_fake_aiter_mla(monkeypatch, mla_decode_fwd)

    assert aiter_ops._check_aiter_mla_fp8_support() is True
    assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is True


def test_aiter_mla_fp8_support_requires_both_scale_parameters(monkeypatch):
    """Partial FP8 signatures must not be treated as full support."""
    import vllm._aiter_ops as aiter_ops

    def mla_decode_fwd(*args, q_scale=None, **kwargs):
        return None

    _install_fake_aiter_mla(monkeypatch, mla_decode_fwd)

    assert aiter_ops._check_aiter_mla_fp8_support() is False
    assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False


def test_aiter_mla_fp8_support_returns_false_when_import_fails(monkeypatch):
    """Missing ``aiter.mla`` should degrade cleanly to the unsupported path."""
    import vllm._aiter_ops as aiter_ops

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "aiter.mla":
            raise ImportError("aiter.mla is unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "aiter", raising=False)
    monkeypatch.delitem(sys.modules, "aiter.mla", raising=False)

    assert aiter_ops._check_aiter_mla_fp8_support() is False
    assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False


@pytest.mark.parametrize(
    "exc",
    [
        AttributeError("mla_decode_fwd is missing"),
        ValueError("mla_decode_fwd has no signature"),
        TypeError("mla_decode_fwd is not callable"),
    ],
    ids=["attribute_error", "value_error", "type_error"],
)
def test_aiter_mla_fp8_support_returns_false_when_signature_probe_fails(
    monkeypatch,
    exc,
):
    """Bad or uninspectable AITER callables should not break decode setup."""
    import vllm._aiter_ops as aiter_ops

    def mla_decode_fwd(*args, **kwargs):
        return None

    def raise_probe_error(*args, **kwargs):
        raise exc

    _install_fake_aiter_mla(monkeypatch, mla_decode_fwd)
    monkeypatch.setattr(inspect, "signature", raise_probe_error)

    assert aiter_ops._check_aiter_mla_fp8_support() is False
    assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False


@pytest.mark.parametrize(
    "cached_result",
    [False, True],
    ids=["cached_false", "cached_true"],
)
def test_aiter_mla_fp8_support_uses_cached_result(monkeypatch, cached_result):
    """Once cached, the support probe should not re-inspect AITER every call."""
    import vllm._aiter_ops as aiter_ops

    aiter_ops._AITER_MLA_SUPPORTS_FP8 = cached_result

    def fail_if_called(*args, **kwargs):
        raise AssertionError("cached support check should not inspect again")

    monkeypatch.setattr(inspect, "signature", fail_if_called)

    assert aiter_ops._check_aiter_mla_fp8_support() is cached_result


def test_rocm_aiter_mla_decode_passes_fp8_scales_and_decode_args_when_supported(
    monkeypatch,
):
    """When FP8 MLA is supported, vLLM must forward scales and decode args."""
    import vllm._aiter_ops as aiter_ops

    captured_args: tuple[object, ...] | None = None
    captured_kwargs: dict[str, object] | None = None

    def mla_decode_fwd(*args: object, **kwargs: object) -> None:
        nonlocal captured_args, captured_kwargs
        captured_args = args
        captured_kwargs = kwargs

    _install_fake_aiter_mla(monkeypatch, mla_decode_fwd)
    monkeypatch.setattr(aiter_ops, "_check_aiter_mla_fp8_support", lambda: True)

    inputs = _make_decode_inputs()
    aiter_ops._rocm_aiter_mla_decode_fwd_impl(**inputs)

    assert captured_args is not None
    assert captured_kwargs is not None
    assert captured_args[0] is inputs["q"]
    assert captured_args[2] is inputs["o"]
    assert captured_args[3] is inputs["qo_indptr"]
    assert captured_args[4] is inputs["kv_indptr"]
    assert captured_args[5] is inputs["kv_indices"]
    assert captured_args[6] is inputs["kv_last_page_lens"]
    assert captured_args[7] == inputs["max_seqlen_qo"]
    kv_arg = captured_args[1]
    assert isinstance(kv_arg, torch.Tensor)
    assert kv_arg.shape == (
        inputs["kv_buffer"].shape[0],
        1,
        1,
        inputs["q"].shape[-1],
    )
    assert captured_kwargs["q_scale"] is inputs["q_scale"]
    assert captured_kwargs["kv_scale"] is inputs["kv_scale"]


def test_rocm_aiter_mla_decode_omits_fp8_scales_when_not_supported(monkeypatch):
    """Older AITER builds must receive decode args without FP8 scale kwargs."""
    import vllm._aiter_ops as aiter_ops

    captured_kwargs: dict[str, object] | None = None

    def mla_decode_fwd(*args: object, **kwargs: object) -> None:
        nonlocal captured_kwargs
        captured_kwargs = kwargs

    _install_fake_aiter_mla(monkeypatch, mla_decode_fwd)
    monkeypatch.setattr(aiter_ops, "_check_aiter_mla_fp8_support", lambda: False)

    inputs = _make_decode_inputs()
    aiter_ops._rocm_aiter_mla_decode_fwd_impl(**inputs)

    assert captured_kwargs is not None
    assert "q_scale" not in captured_kwargs
    assert "kv_scale" not in captured_kwargs


def test_rocm_aiter_mla_decode_passes_work_buffers_when_provided(monkeypatch):
    """Decode should forward MLA work buffers once the backend allocates them."""
    import vllm._aiter_ops as aiter_ops

    captured_kwargs: dict[str, object] | None = None

    def mla_decode_fwd(*args: object, **kwargs: object) -> None:
        nonlocal captured_kwargs
        captured_kwargs = kwargs

    _install_fake_aiter_mla(monkeypatch, mla_decode_fwd)
    monkeypatch.setattr(aiter_ops, "_check_aiter_mla_fp8_support", lambda: True)

    inputs = _make_decode_inputs()
    work_buffers = _make_work_buffer_inputs()
    aiter_ops._rocm_aiter_mla_decode_fwd_impl(**inputs, **work_buffers)

    assert captured_kwargs is not None
    for name, tensor in work_buffers.items():
        assert captured_kwargs[name] is tensor


@pytest.mark.parametrize(
    ("missing_name", "expected_message"),
    [
        ("work_indptr", "work_indptr must be provided with work_meta_data"),
        ("work_info_set", "work_info_set must be provided with work_meta_data"),
        ("reduce_indptr", "reduce_indptr must be provided with work_meta_data"),
        ("reduce_final_map", "reduce_final_map must be provided with work_meta_data"),
        (
            "reduce_partial_map",
            "reduce_partial_map must be provided with work_meta_data",
        ),
    ],
)
def test_rocm_aiter_mla_decode_requires_complete_work_buffer_set(
    monkeypatch,
    missing_name,
    expected_message,
):
    """Partial work-buffer sets should fail early with a clear contract error."""
    import vllm._aiter_ops as aiter_ops

    def mla_decode_fwd(*args, **kwargs):
        raise AssertionError("mla_decode_fwd should not be called")

    _install_fake_aiter_mla(monkeypatch, mla_decode_fwd)
    monkeypatch.setattr(aiter_ops, "_check_aiter_mla_fp8_support", lambda: True)

    inputs = _make_decode_inputs()
    work_buffers = dict(_make_work_buffer_inputs())
    work_buffers.pop(missing_name)

    with pytest.raises(AssertionError, match=expected_message):
        aiter_ops._rocm_aiter_mla_decode_fwd_impl(**inputs, **work_buffers)
