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


def test_missing_causal_arg_fails_closed(monkeypatch):
    """Old aiter wheels have no causal=; do not report that as a valid draft path."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    _install_fake_aiter_modules(monkeypatch, supports_fp8=True, supports_causal=False)
    with pytest.raises(RuntimeError, match="causal-only"):
        rocm_aiter_ops.mla_decode_supports_non_causal()
    with pytest.raises(RuntimeError, match="causal-only"):
        AiterMLABackend.supports_non_causal()


def test_supports_non_causal_requires_gfx950_kernels(monkeypatch):
    """causal= on gfx942 is not a kernel; the backend must fall through."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends.mla import rocm_aiter_mla
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    _install_fake_aiter_modules(monkeypatch, supports_fp8=True, supports_causal=True)
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_non_causal_asm_kernels", lambda: False
    )
    assert rocm_aiter_ops.mla_decode_supports_non_causal() is True
    assert AiterMLABackend.supports_non_causal() is False


def test_supports_non_causal_accepts_gfx950_with_causal(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends.mla import rocm_aiter_mla
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    _install_fake_aiter_modules(monkeypatch, supports_fp8=True, supports_causal=True)
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_non_causal_asm_kernels", lambda: True
    )
    assert rocm_aiter_ops.mla_decode_supports_non_causal() is True
    assert AiterMLABackend.supports_non_causal() is True


@pytest.mark.skipif(
    _SKIP_UNSUPPORTED_AITER_HARDWARE,
    reason="Installed AITER MLA causal= check requires CDNA 3 or newer",
)
def test_installed_aiter_mla_decode_accepts_causal():
    """Supported ROCm CI must ship an aiter whose MLA decode takes causal=.

    Non-causal backend selection also needs the gfx950 ASM kernels; gfx942
    still has the Python argument but no causal=0 decode entries.
    """
    from vllm._aiter_ops import (
        is_aiter_found_and_supported,
        rocm_aiter_ops,
    )
    from vllm.platforms.rocm import on_gfx950
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    assert is_aiter_found_and_supported()
    assert rocm_aiter_ops.mla_decode_supports_non_causal() is True
    assert AiterMLABackend.supports_non_causal() is on_gfx950()


@pytest.mark.parametrize("causal", [True, False])
def test_decode_fwd_forwards_causal_to_aiter(monkeypatch, causal):
    """The decode op must pass the block mask through; omitting it is causal-only."""
    seen: list = []

    def mla_decode_fwd(*args, causal=None, **kwargs):
        seen.append(causal)

    mla_mod: Any = types.ModuleType("aiter.mla")
    mla_mod.mla_decode_fwd = mla_decode_fwd
    aiter_mod: Any = types.ModuleType("aiter")
    aiter_mod.mla = mla_mod
    monkeypatch.setitem(sys.modules, "aiter", aiter_mod)
    monkeypatch.setitem(sys.modules, "aiter.mla", mla_mod)

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
    assert seen == [causal]


def _aiter_mla_validate(
    monkeypatch,
    *,
    dtype,
    kv_cache_dtype,
    use_non_causal,
):
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.mla import rocm_aiter_mla
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    _install_fake_aiter_modules(monkeypatch, supports_fp8=True, supports_causal=True)
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_non_causal_asm_kernels", lambda: True
    )
    return AiterMLABackend.validate_configuration(
        head_size=576,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=1,
        use_mla=True,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 5),
        attn_type=AttentionType.DECODER,
        use_non_causal=use_non_causal,
    )


def test_non_causal_fp16_auto_cache_is_rejected(monkeypatch):
    """Pinned AITER aborts fp16 Q; auto-select must fall through to Triton."""
    import torch

    reasons = _aiter_mla_validate(
        monkeypatch,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        use_non_causal=True,
    )
    assert any("non-causal fp16" in reason for reason in reasons)


def test_non_causal_bf16_auto_cache_is_accepted(monkeypatch):
    import torch

    reasons = _aiter_mla_validate(
        monkeypatch,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        use_non_causal=True,
    )
    assert reasons == []


def test_causal_fp16_auto_cache_is_still_accepted(monkeypatch):
    import torch

    reasons = _aiter_mla_validate(
        monkeypatch,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        use_non_causal=False,
    )
    assert reasons == []
