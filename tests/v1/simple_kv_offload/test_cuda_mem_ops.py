# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for resolving cuMemcpyBatchAsync in
vllm.v1.simple_kv_offload.cuda_mem_ops.

cuda.bindings.driver is replaced by a fake module, so nothing here needs a GPU
or a particular driver.
"""

import enum
import sys
import types
from typing import Any

import pytest

from vllm.v1.simple_kv_offload import cuda_mem_ops


class _CUresult(enum.IntEnum):
    CUDA_SUCCESS = 0
    CUDA_ERROR_NOT_INITIALIZED = 3


class _QueryResult(enum.IntEnum):
    CU_GET_PROC_ADDRESS_SUCCESS = 0
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2


def _install_fake_driver(monkeypatch, result, driver_version=12040):
    drv = types.SimpleNamespace(
        CUresult=_CUresult,
        CUdriverProcAddressQueryResult=_QueryResult,
        cuGetProcAddress=lambda name, version, flags: result,
        cuDriverGetVersion=lambda: (_CUresult.CUDA_SUCCESS, driver_version),
    )
    bindings: Any = types.ModuleType("cuda.bindings")
    bindings.driver = drv
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", drv)
    monkeypatch.setattr(cuda_mem_ops.current_platform, "is_rocm", lambda: False)


@pytest.mark.parametrize(
    "status",
    [
        _QueryResult.CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND,
        _QueryResult.CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT,
    ],
)
def test_missing_symbol_raises_instead_of_returning_null(monkeypatch, status):
    # What a CUDA 12.4 driver (e.g. 550.x) returns: success, NULL, not found.
    _install_fake_driver(monkeypatch, (_CUresult.CUDA_SUCCESS, 0, status))
    with pytest.raises(RuntimeError, match="requires a driver supporting CUDA 12.8"):
        cuda_mem_ops._resolve_batch_memcpy()


def test_null_pointer_with_success_status_raises(monkeypatch):
    _install_fake_driver(
        monkeypatch,
        (_CUresult.CUDA_SUCCESS, 0, _QueryResult.CU_GET_PROC_ADDRESS_SUCCESS),
    )
    with pytest.raises(RuntimeError, match="not available"):
        cuda_mem_ops._resolve_batch_memcpy()


def test_lookup_error_raises(monkeypatch):
    _install_fake_driver(
        monkeypatch,
        (
            _CUresult.CUDA_ERROR_NOT_INITIALIZED,
            0,
            _QueryResult.CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND,
        ),
    )
    with pytest.raises(RuntimeError, match="failed"):
        cuda_mem_ops._resolve_batch_memcpy()


def test_found_symbol_is_returned(monkeypatch):
    sentinel = 0x1234
    _install_fake_driver(
        monkeypatch,
        (_CUresult.CUDA_SUCCESS, sentinel, _QueryResult.CU_GET_PROC_ADDRESS_SUCCESS),
        driver_version=12080,
    )
    fn, num_attrs = cuda_mem_ops._resolve_batch_memcpy()
    assert num_attrs == 1
    assert cuda_mem_ops.ctypes.cast(fn, cuda_mem_ops.ctypes.c_void_p).value == sentinel
