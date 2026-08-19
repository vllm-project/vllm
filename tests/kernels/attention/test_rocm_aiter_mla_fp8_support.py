# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ROCm AITER MLA FP8 support detection."""

import sys
import types
from typing import Any
from unittest.mock import patch

import pytest

from vllm.platforms import current_platform

_SKIP_UNSUPPORTED_AITER_HARDWARE = True
if current_platform.is_rocm():
    from vllm.platforms.rocm import get_cdna_version

    _SKIP_UNSUPPORTED_AITER_HARDWARE = get_cdna_version() <= 2

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


@pytest.fixture(autouse=True)
def reset_aiter_mla_support_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm._aiter_ops as aiter_ops

    monkeypatch.setattr(aiter_ops, "_AITER_MLA_SUPPORTS_FP8", None)
    # if_aiter_supported wraps the functools.cache, so reach through to it.
    aiter_ops.rocm_aiter_ops.mla_decode_supports_non_causal.__wrapped__.cache_clear()


def _install_fake_aiter_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    supports_fp8: bool,
    supports_causal: bool = False,
) -> None:
    aiter_mod: Any = types.ModuleType("aiter")
    mla_mod: Any = types.ModuleType("aiter.mla")

    if supports_fp8:

        def mla_decode_fwd_with_fp8(
            q,
            kv_buffer,
            kv_indptr,
            kv_indices,
            o,
            sm_scale,
            q_scale=None,
            kv_scale=None,
        ):
            return None

        mla_decode_fwd: Any = mla_decode_fwd_with_fp8

    else:

        def mla_decode_fwd_without_fp8(
            q,
            kv_buffer,
            kv_indptr,
            kv_indices,
            o,
            sm_scale,
        ):
            return None

        mla_decode_fwd = mla_decode_fwd_without_fp8

    if supports_causal:
        inner = mla_decode_fwd

        def mla_decode_fwd_with_causal(*args, causal=True, **kwargs):
            return inner(*args, **kwargs)

        mla_decode_fwd = mla_decode_fwd_with_causal

    mla_mod.mla_decode_fwd = mla_decode_fwd
    aiter_mod.mla = mla_mod

    monkeypatch.setitem(sys.modules, "aiter", aiter_mod)
    monkeypatch.setitem(sys.modules, "aiter.mla", mla_mod)


def test_aiter_mla_fp8_support_detects_fp8_signature(monkeypatch):
    """The support check should detect q_scale and kv_scale parameters."""
    from vllm._aiter_ops import _check_aiter_mla_fp8_support

    _install_fake_aiter_modules(monkeypatch, supports_fp8=True)

    assert _check_aiter_mla_fp8_support() is True


def test_aiter_mla_fp8_support_rejects_missing_fp8_signature(monkeypatch):
    """The support check should return False when FP8 params are absent."""
    from vllm._aiter_ops import _check_aiter_mla_fp8_support

    _install_fake_aiter_modules(monkeypatch, supports_fp8=False)

    assert _check_aiter_mla_fp8_support() is False


@pytest.mark.skipif(
    _SKIP_UNSUPPORTED_AITER_HARDWARE,
    reason="Installed AITER MLA FP8 check requires CDNA 3 or newer",
)
def test_installed_aiter_mla_supports_fp8():
    """Supported ROCm CI must provide AITER with MLA FP8 scaling."""
    from vllm._aiter_ops import (
        _check_aiter_mla_fp8_support,
        is_aiter_found_and_supported,
    )

    assert is_aiter_found_and_supported(), (
        "AITER must be installed on supported ROCm hardware"
    )
    assert _check_aiter_mla_fp8_support() is True


@pytest.mark.parametrize(
    "error_type",
    [ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError],
)
def test_aiter_mla_fp8_support_handles_signature_errors(monkeypatch, error_type):
    """The support check should fail closed on import or signature problems."""
    import vllm._aiter_ops as aiter_ops
    from vllm._aiter_ops import _check_aiter_mla_fp8_support

    _install_fake_aiter_modules(monkeypatch, supports_fp8=True)

    with patch("inspect.signature", side_effect=error_type("boom")):
        assert _check_aiter_mla_fp8_support() is False
        assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False


def test_aiter_mla_fp8_support_result_is_cached(monkeypatch):
    """The support check should reuse the cached result on later calls."""
    import inspect

    from vllm._aiter_ops import _check_aiter_mla_fp8_support

    _install_fake_aiter_modules(monkeypatch, supports_fp8=True)

    with patch("inspect.signature", wraps=inspect.signature) as signature_mock:
        assert _check_aiter_mla_fp8_support() is True
        assert _check_aiter_mla_fp8_support() is True
        assert signature_mock.call_count == 1


@pytest.mark.parametrize("supports_causal", [True, False])
def test_non_causal_probe_follows_the_installed_signature(monkeypatch, supports_causal):
    """Builds without a ``causal`` argument are causal-only; asking one for a
    non-causal block would return a causally masked result and say nothing."""
    from vllm._aiter_ops import rocm_aiter_ops

    _install_fake_aiter_modules(
        monkeypatch, supports_fp8=True, supports_causal=supports_causal
    )
    assert bool(rocm_aiter_ops.mla_decode_supports_non_causal()) is supports_causal


def _install_callable_fake(monkeypatch) -> list:
    """The shared fake above exists to be inspected, not called. The impl tests
    need one that runs, so they bring their own."""
    seen: list = []

    def mla_decode_fwd(*args, causal=None, **kwargs):
        seen.append(
            {"causal": causal, **kwargs} if causal is not None else dict(kwargs)
        )

    mla_mod: Any = types.ModuleType("aiter.mla")
    mla_mod.mla_decode_fwd = mla_decode_fwd
    aiter_mod: Any = types.ModuleType("aiter")
    aiter_mod.mla = mla_mod
    monkeypatch.setitem(sys.modules, "aiter", aiter_mod)
    monkeypatch.setitem(sys.modules, "aiter.mla", mla_mod)
    return seen


def _call_impl(causal: bool) -> None:
    import torch

    import vllm._aiter_ops as aiter_ops

    aiter_ops._rocm_aiter_mla_decode_fwd_impl(
        torch.zeros(2, 1, 8),
        torch.zeros(2, 1, 1, 8),
        torch.zeros(2, 1, 8),
        torch.zeros(3, dtype=torch.int32),
        1,
        causal=causal,
    )


def test_a_causal_block_never_passes_the_argument(monkeypatch):
    """Causal is the default, so builds without the argument keep working."""
    seen = _install_callable_fake(monkeypatch)
    _call_impl(causal=True)
    assert "causal" not in seen[0]


def test_a_non_causal_block_reaches_a_capable_build(monkeypatch):
    seen = _install_callable_fake(monkeypatch)
    _call_impl(causal=False)
    assert seen[0]["causal"] is False
