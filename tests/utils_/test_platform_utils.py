# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CUDA driver/toolkit helpers in vllm.utils.platform_utils.

All CUDA driver queries are mocked, so these run without a GPU.
"""

import ctypes
from unittest.mock import patch

import pytest

import vllm.utils.platform_utils as pu


@pytest.fixture(autouse=True)
def _clear_caches():
    pu.get_cuda_driver_version.cache_clear()
    pu.warn_if_cuda_driver_cannot_jit_ptx.cache_clear()
    yield
    pu.get_cuda_driver_version.cache_clear()
    pu.warn_if_cuda_driver_cannot_jit_ptx.cache_clear()


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("12.9", (12, 9)),
        ("12.8.93", (12, 8)),
        ("13.0", (13, 0)),
        (None, None),
        ("", None),
        ("12", None),
        ("abc.def", None),
    ],
)
def test_parse_cuda_version(version, expected):
    assert pu.parse_cuda_version(version) == expected


class _FakeLibCuda:
    def __init__(self, raw_version: int, rc: int = 0):
        self.raw_version = raw_version
        self.rc = rc

    def cuDriverGetVersion(self, ptr):  # noqa: N802 - mirrors the CUDA API
        ptr._obj.value = self.raw_version
        return self.rc


@pytest.mark.parametrize(
    ("raw", "expected"), [(12080, (12, 8)), (12090, (12, 9)), (13020, (13, 2))]
)
def test_get_cuda_driver_version(raw, expected):
    with patch.object(pu.ctypes, "CDLL", return_value=_FakeLibCuda(raw)):
        assert pu.get_cuda_driver_version() == expected


def test_get_cuda_driver_version_no_libcuda():
    with patch.object(pu.ctypes, "CDLL", side_effect=OSError("no libcuda")):
        assert pu.get_cuda_driver_version() is None


def test_get_cuda_driver_version_call_fails():
    with patch.object(pu.ctypes, "CDLL", return_value=_FakeLibCuda(12080, rc=3)):
        assert pu.get_cuda_driver_version() is None


def test_fake_libcuda_matches_ctypes_byref():
    # Guard the fake above: byref(c_int)._obj is the c_int being written.
    v = ctypes.c_int(0)
    _FakeLibCuda(12080).cuDriverGetVersion(ctypes.byref(v))
    assert v.value == 12080


def _run_check(monkeypatch, driver, toolkit):
    monkeypatch.setattr(pu, "get_cuda_driver_version", lambda: driver)
    monkeypatch.setattr(pu.torch.version, "cuda", toolkit)
    with patch.object(pu.logger, "warning") as mock_warn:
        result = pu.warn_if_cuda_driver_cannot_jit_ptx("Marlin")
    return result, mock_warn


def test_warns_when_driver_older_than_toolkit(monkeypatch):
    result, mock_warn = _run_check(monkeypatch, (12, 8), "12.9")
    assert result is True
    mock_warn.assert_called_once()
    fmt, *args = mock_warn.call_args[0]
    message = fmt % tuple(args)
    assert "CUDA 12.8" in message
    assert "built with CUDA 12.9" in message
    assert "Marlin" in message
    assert "unsupported toolchain" in message
    assert "VLLM_ENABLE_CUDA_COMPATIBILITY=1" in message


@pytest.mark.parametrize(
    ("driver", "toolkit"),
    [
        ((12, 9), "12.9"),  # match
        ((13, 0), "12.9"),  # newer driver
        (None, "12.9"),  # driver unknown
        ((12, 8), None),  # not a CUDA build of torch
    ],
)
def test_no_warning_without_mismatch(monkeypatch, driver, toolkit):
    result, mock_warn = _run_check(monkeypatch, driver, toolkit)
    assert result is False
    mock_warn.assert_not_called()


def test_warns_only_once_per_kernel(monkeypatch):
    monkeypatch.setattr(pu, "get_cuda_driver_version", lambda: (12, 8))
    monkeypatch.setattr(pu.torch.version, "cuda", "12.9")
    with patch.object(pu.logger, "warning") as mock_warn:
        for _ in range(5):
            pu.warn_if_cuda_driver_cannot_jit_ptx("Marlin")
    mock_warn.assert_called_once()
