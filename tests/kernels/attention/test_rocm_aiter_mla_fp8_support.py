# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ROCm AITER MLA FP8 support detection."""

import sys
import types
from typing import Any
from unittest.mock import patch

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


def _reset_aiter_mla_support_cache() -> None:
    import vllm._aiter_ops as aiter_ops

    aiter_ops._AITER_MLA_SUPPORTS_FP8 = None


def _install_fake_aiter_modules(
    monkeypatch: pytest.MonkeyPatch, *, supports_fp8: bool
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

    mla_mod.mla_decode_fwd = mla_decode_fwd
    aiter_mod.mla = mla_mod

    monkeypatch.setitem(sys.modules, "aiter", aiter_mod)
    monkeypatch.setitem(sys.modules, "aiter.mla", mla_mod)


def test_aiter_mla_fp8_support_detects_fp8_signature(monkeypatch):
    """The support check should detect q_scale and kv_scale parameters."""
    from vllm._aiter_ops import _check_aiter_mla_fp8_support

    _reset_aiter_mla_support_cache()
    _install_fake_aiter_modules(monkeypatch, supports_fp8=True)

    assert _check_aiter_mla_fp8_support() is True


def test_aiter_mla_fp8_support_rejects_missing_fp8_signature(monkeypatch):
    """The support check should return False when FP8 params are absent."""
    from vllm._aiter_ops import _check_aiter_mla_fp8_support

    _reset_aiter_mla_support_cache()
    _install_fake_aiter_modules(monkeypatch, supports_fp8=False)

    assert _check_aiter_mla_fp8_support() is False


@pytest.mark.parametrize(
    "error_type",
    [ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError],
)
def test_aiter_mla_fp8_support_handles_signature_errors(monkeypatch, error_type):
    """The support check should fail closed on import or signature problems."""
    import vllm._aiter_ops as aiter_ops
    from vllm._aiter_ops import _check_aiter_mla_fp8_support

    _reset_aiter_mla_support_cache()
    _install_fake_aiter_modules(monkeypatch, supports_fp8=True)

    with patch("inspect.signature", side_effect=error_type("boom")):
        assert _check_aiter_mla_fp8_support() is False
        assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False


def test_aiter_mla_fp8_support_result_is_cached(monkeypatch):
    """The support check should reuse the cached result on later calls."""
    import inspect

    from vllm._aiter_ops import _check_aiter_mla_fp8_support

    _reset_aiter_mla_support_cache()
    _install_fake_aiter_modules(monkeypatch, supports_fp8=True)

    with patch("inspect.signature", wraps=inspect.signature) as signature_mock:
        assert _check_aiter_mla_fp8_support() is True
        assert _check_aiter_mla_fp8_support() is True
        assert signature_mock.call_count == 1
